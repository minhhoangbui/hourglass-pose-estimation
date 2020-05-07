from __future__ import print_function

import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import datetime
import re
from torch.utils.tensorboard import SummaryWriter

import _init_path
from pose import Bar
from pose.utils.logger import Logger
from pose.utils.evaluation import accuracy, AverageMeter
from pose.utils.misc import save_checkpoint, adjust_learning_rate
import pose.models as models
import pose.datasets as datasets
from pose.loss.loss import JointsMSELoss
from torchsummary import summary


# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))

# init global variables
best_acc = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def train(train_loader, model, criterion, optimizer, pck, idxs=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()

    end = time.time()

    bar = Bar('Training', max=len(train_loader))

    for i, (inputs, target, meta) in enumerate(train_loader):
        if idxs is not None:
            target = torch.index_select(target, 1, idxs)
        data_time.update(time.time() - end)

        inputs, target = inputs.to(device), target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        # compute output
        output = model(inputs)
        if type(output) == list:  # multiple output
            loss = 0
            for o in output:
                loss += criterion(o, target, target_weight)
            score_map = output[-1]
        else:
            raise ValueError('Format failed!!!')
        acc = accuracy(score_map, target, idxs=idxs, thr=pck)

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ' \
                     'ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            acc=acces.avg
                        )
        bar.next()
    bar.finish()
    return losses.avg, acces.avg


def validate(val_loader, model, criterion, pck, idxs=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Evaluating ', max=len(val_loader))
    with torch.no_grad():
        for i, (inputs, target, meta) in enumerate(val_loader):
            if idxs is not None:
                target = torch.index_select(target, 1, idxs)

            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)
            output = model(inputs)

            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target, target_weight)
                score_map = output[-1].cpu()
            else:  # single output
                raise ValueError('Format failed!!!')

            acc = accuracy(score_map, target.cpu(), idxs=idxs, thr=pck)

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            acces.update(acc[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                                batch=i + 1,
                                size=len(val_loader),
                                data=data_time.val,
                                bt=batch_time.avg,
                                total=bar.elapsed_td,
                                eta=bar.eta_td,
                                loss=losses.avg,
                                acc=acces.avg
                            )
            bar.next()
        bar.finish()
    return losses.avg, acces.avg


def main(args):
    global best_acc
    idxs_str = str(args.subset)
    idxs_str = idxs_str.strip('[]')
    idxs_str = re.sub(r'\s+', '', idxs_str)
    checkpoint_path = os.path.join(args.checkpoint,
                                   '{}_s{}_b{}_{}_{}_{}'.format(args.dataset, args.stacks,
                                                                args.blocks, 'mobile' if args.mobile else 'non-mobile',
                                                                'all' if args.subset is None else idxs_str,
                                                                args.skip_mode))

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # create model

    n_joints = datasets.__dict__[args.dataset].n_joints if args.subset is None else \
        len(args.subset)

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=n_joints,
                                       mobile=args.mobile,
                                       skip_mode=args.skip_mode)
    summary(model, (3, args.inp_res, args.inp_res), device='cpu')
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'tensorboard'))

    model = torch.nn.DataParallel(model).to(device)

    criterion = JointsMSELoss(use_target_weight=True)
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = args.dataset + ' ' + args.arch

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=True)
        else:
            raise OSError("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if os.path.isfile(os.path.join(checkpoint_path, 'log.txt')):
            logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=False)
        logger.set_names(['Time', 'Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])

    # create data loader
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    idxs = args.subset
    if args.subset is not None:
        idxs = torch.LongTensor(args.subset)

    # evaluation only
    if args.evaluate:
        print('\nEvaluation only')
        _, _ = validate(val_loader=val_loader, model=model,
                        criterion=criterion, pck=args.pck,
                        idxs=idxs)
        return

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *= args.sigma_decay
            val_loader.dataset.sigma *= args.sigma_decay

        # train for one epoch
        train_loss, train_acc = train(train_loader=train_loader, model=model, criterion=criterion,
                                      optimizer=optimizer, idxs=idxs, pck=args.pck)

        # evaluate on validation set
        valid_loss, valid_acc = validate(val_loader=val_loader, model=model,
                                         criterion=criterion, idxs=idxs,
                                         pck=args.pck)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', valid_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/test', valid_acc, epoch)

        # append logger file
        logger.append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), epoch + 1, lr, train_loss, valid_loss,
                       train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path, snapshot=args.snapshot)

    writer.close()
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Pose Estimation Training')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='mpii',
                        choices=dataset_names,
                        help='Datasets: ' +
                             ' | '.join(dataset_names) +
                             ' (default: mpii)')
    parser.add_argument('--image-path', default='', type=str,
                        help='path to images')
    parser.add_argument('--anno-path', default='', type=str,
                        help='path to annotation (json)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                        help='output resolution (default: 64, to gen GT)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--mobile', action='store_true',
                        help='Decide to use mobile architecture')
    parser.add_argument('--skip_mode', default='sum', type=str,
                        help='Decide which skip_mode to use in residual architecture')
    parser.add_argument('--subset', type=int, nargs='+', default=None,
                        help='Decide subset when training or evaluating')

    # training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')

    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--scale-factor', type=float, default=0.25,
                        help='Scale factor (data aug).')
    parser.add_argument('--rot-factor', type=float, default=30,
                        help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')

    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pck', type=float, default=0.3,
                        help='evaluate model on validation set')

    main(parser.parse_args())


