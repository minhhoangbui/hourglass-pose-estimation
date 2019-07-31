from __future__ import print_function

import os
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import _init_path
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.imutils import batch_with_heatmap
import pose.models as models
import pose.datasets as datasets

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


def load_teacher_models(arch, blocks, stacks, t_checkpoint, num_classes):
    tmodel = models.__dict__[arch](num_stacks=stacks, num_blocks=blocks,
                                   num_classes=num_classes)
    tmodel = torch.nn.DataParallel(tmodel).cuda()

    t_checkpoint = torch.load(t_checkpoint)
    tmodel.load_state_dict(t_checkpoint['state_dict'])
    tmodel.eval()
    return tmodel


def train(train_loader, model, t_model, criterion, optimizer, kdloss_alpha, debug=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    gt_losses = AverageMeter()
    acces = AverageMeter()

    model.train()

    end = time.time()

    gt_win, pred_win, pred_teacher = None, None, None
    bar = Bar('Training', max=len(train_loader))

    for i, (inputs, target, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs, target = inputs.to(device), target.to(device, non_blocking=True)

        # compute output
        output = model(inputs)
        score_map = output[-1]

        t_output = t_model(inputs)
        t_output = t_output[-1].detach()  # 4D [batch,  num_classes, out_res, out_res]

        gt_loss = torch.tensor(0.0).cuda()
        kd_loss = torch.tensor(0.0).cuda()

        for j in range(0, len(output)):
            _output = output[j]
            kd_loss += criterion(_output, t_output)
            gt_loss += criterion(_output, target)
        acc = accuracy(score_map, target)
        # This is confirmed from the source code of the authors. This is weighted sum of loss from each heat-map
        total_loss = kdloss_alpha * kd_loss + (1.0 - kdloss_alpha) * gt_loss

        if debug:  # visualize ground-truth and predictions
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, output)
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        kd_losses.update(kd_loss.item(), inputs.size(0))
        gt_losses.update(gt_loss.item(), inputs.size(0))
        losses.update(total_loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
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


def validate(val_loader, model, t_model, criterion, num_classes, kdloss_alpha, out_res=64, debug=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Evaluating', max=len(val_loader))
    with torch.no_grad():
        for i, (inputs, target, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            t_output = t_model(inputs)[-1].detach()

            output = model(inputs)
            score_map = output[-1].cpu() if type(output) == list else output.cpu()

            gt_loss = torch.tensor(0.0).cuda()
            kd_loss = torch.tensor(0.0).cuda()

            for j in range(0, len(output)):
                _output = output[j]
                kd_loss += criterion(_output, t_output)
                gt_loss += criterion(_output, target)

            total_loss = kdloss_alpha * kd_loss + (1.0 - kdloss_alpha) * gt_loss

            acc = accuracy(score_map, target.cpu())

            preds = final_preds(score_map, meta['center'], meta['scale'], [out_res, out_res])
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            if debug:
                gt_batch_img = batch_with_heatmap(inputs, target)
                pred_batch_img = batch_with_heatmap(inputs, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    gt_win = plt.imshow(gt_batch_img)
                    plt.subplot(122)
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            # measure accuracy and record loss
            losses.update(total_loss.item(), inputs.size(0))
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
    return losses.avg, acces.avg, predictions


def main(args):
    global best_acc
    global idx

    if args.dataset in ['mpii', 'lsp']:
        idx = [1, 2, 3, 4, 5, 6, 11, 12, 15, 16]
    else:
        raise ValueError('Unsupported dataset')

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    # create model
    njoints = datasets.__dict__[args.dataset].njoints

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=njoints)

    print("==> creating teacher model '{}', stacks={}, blocks={}".format(args.arch, args.teacher_stacks, args.blocks))
    tmodel = load_teacher_models(arch=args.arch, blocks=args.blocks, stacks=args.teacher_stacks,
                                 t_checkpoint=args.teacher_checkpoint, num_classes=njoints)

    model = torch.nn.DataParallel(model).to(device)
    criterion = torch.nn.MSELoss(size_average=True).cuda()
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
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters()) / 1000000.0))

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

    # evaluation only
    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader=val_loader, model=model, t_model=tmodel,
                                          criterion=criterion, num_classes=njoints,
                                          kdloss_alpha=args.kdloss_alpha, out_res=args.out_res, debug=args.debug)
        save_pred(predictions, checkpoint=args.checkpoint)
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
        train_loss, train_acc = train(train_loader=train_loader, model=model, t_model=tmodel, criterion=criterion,
                                      kdloss_alpha=args.kdloss_alpha, optimizer=optimizer, debug=args.debug)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = validate(val_loader=val_loader, model=model, t_model=tmodel,
                                                      criterion=criterion, num_classes=njoints,
                                                      kdloss_alpha=args.kdloss_alpha, out_res=args.out_res,
                                                      debug=args.debug)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--dataset', metavar='DATASET', default='mpii',
                        choices=dataset_names,
                        help='Datasets: ' +
                             ' | '.join(dataset_names) +
                             ' (default: mpii)')
    parser.add_argument('--image-path', default='', type=str,
                        help='path to images')
    parser.add_argument('--anno-path', default='', type=str,
                        help='path to annotation (json)')

    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                        help='output resolution (default: 64, to gen GT)')
    parser.add_argument("--teacher-stacks", type=int)

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    # parser.add_argument('--features', default=256, type=int, metavar='N',
    #                     help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')

    # Training strategy
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
    # parser.add_argument('--target-weight', dest='target_weight',
    #                     action='store_true',
    #                     help='Loss with target_weight')
    parser.add_argument('--kdloss-alpha', type=float, default=0.5,
                        help='coefficient for kdloss')

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
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('--teacher-checkpoint', metavar='PATH',
                        help='path to teacher checkpoint')

    main(parser.parse_args())
