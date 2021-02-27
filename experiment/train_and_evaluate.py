from __future__ import print_function

import os
import yaml
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import datetime
import sys
from torch.utils.tensorboard import SummaryWriter

import _init_path
from pose import Bar
from pose.utils.logger import Logger
from pose.utils.evaluation import accuracy, AverageMeter
from pose.utils.misc import save_checkpoint, adjust_learning_rate
import pose.models as models
import pose.datasets as datasets
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
        outputs = model(inputs)
        score_map = outputs[-1]
        loss = criterion(outputs, target, target_weight)
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

            outputs = model(inputs)
            score_map = outputs[-1].cpu()
            loss = criterion(outputs, target, target_weight)

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


def main(cfg):
    global best_acc
    checkpoint_path = os.path.join(cfg['MISC']['checkpoint'],
                                   '{}_{}_s{}_{}_{}_{}'.format(cfg['DATASET']['name'], cfg['MODEL']['arch'],
                                                               cfg['MODEL']['num_stacks'],
                                                               'mobile' if cfg['MODEL']['mobile'] else 'non-mobile',
                                                               'all' if cfg['MODEL']['subset'] is None else cfg['MODEL']['subset'],
                                                               cfg['MODEL']['skip_mode']))

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # create model

    n_joints = datasets.__dict__[cfg['DATASET']['name']].n_joints if cfg['MODEL']['subset'] is None else \
        len(cfg['MODEL']['subset'])

    print("==> creating model '{}', stacks={}, blocks={}".format(cfg['MODEL']['arch'],
                                                                 cfg['MODEL']['num_stacks'],
                                                                 cfg['MODEL']['num_blocks']))
    o_model = models.__dict__[cfg['MODEL']['arch']](num_stacks=cfg['MODEL']['num_stacks'],
                                                    num_blocks=cfg['MODEL']['num_blocks'],
                                                    num_classes=n_joints,
                                                    mobile=cfg['MODEL']['mobile'],
                                                    skip_mode=cfg['MODEL']['skip_mode'],
                                                    out_res=cfg['DATASET']['out_res'])

    summary(o_model, (3, cfg['DATASET']['inp_res'], cfg['DATASET']['inp_res']), device='cpu')
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'tensorboard'))

    model = torch.nn.DataParallel(o_model).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=cfg['TRAIN']['learning_rate'],
                                    momentum=cfg['TRAIN']['momentum'],
                                    weight_decay=cfg['TRAIN']['weight_decay'])

    # optionally resume from a checkpoint
    title = cfg['DATASET']['name'] + ' ' + cfg['MODEL']['arch']

    if cfg['MISC']['resume']:
        if os.path.isfile(cfg['MISC']['resume']):
            print("=> loading checkpoint '{}'".format(cfg['MISC']['resume']))
            checkpoint = torch.load(cfg['MISC']['resume'])
            cfg['TRAIN']['start_epoch'] = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg['MISC']['resume'], checkpoint['epoch']))
            logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=True)
        else:
            raise OSError("=> no checkpoint found at '{}'".format(cfg['MISC']['resume']))
    else:
        if os.path.isfile(os.path.join(checkpoint_path, 'log.txt')):
            logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=False)
        logger.set_names(['Time', 'Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])
    if cfg['MISC']['out_onnx'] and cfg['MISC']['resume']:
        dummy_input = torch.randn(1, 3, cfg['DATASET']['inp_res'], cfg['DATASET']['inp_res'], device='cuda')
        torch.onnx.export(model.module, dummy_input, cfg['MISC']['out_onnx'], opset_version=10)
        exit()

    # create data loader
    train_dataset = datasets.__dict__[cfg['DATASET']['name']](is_train=True, **cfg['DATASET'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['TRAIN']['train_batch'], shuffle=True,
        num_workers=cfg['TRAIN']['num_workers'], pin_memory=True
    )

    val_dataset = datasets.__dict__[cfg['DATASET']['name']](is_train=False, **cfg['DATASET'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg['TRAIN']['test_batch'], shuffle=False,
        num_workers=cfg['TRAIN']['num_workers'], pin_memory=True
    )
    idxs = cfg['MODEL']['subset']
    if idxs is not None:
        idxs = torch.LongTensor(cfg['MODEL']['subset'])

    # evaluation only
    if cfg['MISC']['evaluate']:
        print('\nEvaluation only')
        _, _ = validate(val_loader=val_loader, model=model,
                        criterion=o_model.compute_loss,
                        pck=cfg['MISC']['pck'], idxs=idxs)
        return

    # train and eval
    lr = cfg['TRAIN']['learning_rate']
    for epoch in range(cfg['TRAIN']['start_epoch'], cfg['TRAIN']['epochs']):
        lr = adjust_learning_rate(optimizer, epoch, lr, cfg['TRAIN']['schedule'], cfg['TRAIN']['gamma'])
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if cfg['TRAIN']['sigma_decay']:
            train_loader.dataset.sigma *= cfg['TRAIN']['sigma_decay']
            # val_loader.dataset.sigma *= cfg['TRAIN']['sigma_decay']

        # train for one epoch
        train_loss, train_acc = train(train_loader=train_loader, model=model,
                                      criterion=o_model.compute_loss,
                                      optimizer=optimizer, idxs=idxs,
                                      pck=cfg['MISC']['pck'])

        # evaluate on validation set
        valid_loss, valid_acc = validate(val_loader=val_loader, model=model,
                                         criterion=o_model.compute_loss,
                                         idxs=idxs, pck=cfg['MISC']['pck'])

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
            'arch': cfg['MODEL']['arch'],
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path, snapshot=cfg['MISC']['snapshot'])

    writer.close()
    logger.close()


if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    main(cfg)


