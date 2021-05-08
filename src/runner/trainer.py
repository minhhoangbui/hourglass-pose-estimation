import torch
import torch.optim
import torch.utils.data
import os
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar
import torch.backends.cudnn as cudnn

from src.loss.mse import MSELoss
from src import datasets, models
from src.utils.evaluation import AverageMeter, accuracy


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


class Trainer(object):
    def __init__(self, cfg, num_classes):
        self.cfg = cfg
        print(f"==> creating model '{cfg['MODEL']['arch']}', stacks={cfg['MODEL']['num_stacks']}")
        model = models.__dict__[cfg['MODEL']['arch']](num_stacks=cfg['MODEL']['num_stacks'],
                                                      num_blocks=1,
                                                      num_classes=num_classes,
                                                      mobile=cfg['MODEL']['mobile'],
                                                      skip_mode=cfg['MODEL']['skip_mode'],
                                                      out_res=cfg['DATASET']['out_res'])
        summary(model, (3, cfg['DATASET']['inp_res'], cfg['DATASET']['inp_res']), device='cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cudnn.benchmark = True
        self.model = torch.nn.DataParallel(model).to(self.device)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                             lr=cfg['TRAIN']['learning_rate'],
                                             momentum=0, weight_decay=0)

        self.criterion = MSELoss(use_target_weight=True)
        self.start_epoch = 0
        self.best_acc = 0

        train_dataset = datasets.__dict__[cfg['DATASET']['name']](is_train=True, **cfg['DATASET'])
        val_dataset = datasets.__dict__[cfg['DATASET']['name']](is_train=False, **cfg['DATASET'])
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg['TRAIN']['train_batch'], shuffle=True,
            num_workers=cfg['TRAIN']['num_workers'], pin_memory=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg['TRAIN']['val_batch'], shuffle=True,
            num_workers=cfg['TRAIN']['num_workers'], pin_memory=True
        )

        self.idxs = cfg['MODEL']['subset']
        self.writer = SummaryWriter(log_dir=os.path.join(cfg['COMMON']['checkpoint_dir'], 'logs', 'train'))

        if os.path.isfile(cfg['COMMON']['resume']):
            self._resume()

    def _resume(self):
        print("=> loading checkpoint '{}'".format(self.cfg['COMMON']['resume']))
        checkpoint = torch.load(self.cfg['COMMON']['resume'])
        self.start_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_acc = checkpoint['best_acc']

    def _train_epoch(self):
        self.model.train()
        average_loss = AverageMeter()
        average_acc = AverageMeter()

        bar = Bar('Training', max=len(self.train_loader))

        for i, (images, heatmaps, meta) in enumerate(self.train_loader):
            if self.idxs:
                heatmaps = torch.index_select(heatmaps, 1, torch.LongTensor(self.idxs))
            images = images.to(self.device)
            heatmaps = heatmaps.to(self.device, non_blocking=True)
            target_weight = meta['target_weight'].to(self.device, non_blocking=True)

            outputs = self.model(images)
            last_hms = outputs[-1]
            loss = self.criterion(outputs, heatmaps, target_weight)
            acc = accuracy(last_hms, heatmaps, self.idxs, thr=self.cfg['COMMON']['pck'])

            average_loss.update(loss.item(), images.size(0))
            average_acc.update(acc[0], images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bar.suffix = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                batch=i+1, size=len(self.train_loader), total=bar.elapsed_td, eta=bar.eta_td,
                loss=average_loss.avg, acc=average_acc.avg
            )

            bar.next()
        bar.finish()
        return average_loss.avg, average_acc.avg

    def _evaluate(self):
        self.model.eval()

        average_loss = AverageMeter()
        average_acc = AverageMeter()

        bar = Bar('Evaluating', max=len(self.val_loader))

        for i, (images, heatmaps, meta) in enumerate(self.val_loader):
            if self.idxs:
                heatmaps = torch.index_select(heatmaps, 1, torch.LongTensor(self.idxs))
            images = images.to(self.device)
            heatmaps = heatmaps.to(self.device, non_blocking=True)
            target_weight = meta['target_weight'].to(self.device, non_blocking=True)

            outputs = self.model(images)
            last_hms = outputs[-1]
            loss = self.criterion(outputs, heatmaps, target_weight)
            acc = accuracy(last_hms, heatmaps, self.idxs, thr=self.cfg['COMMON']['pck'])

            average_loss.update(loss.item(), images.size(0))
            average_acc.update(acc[0], images.size(0))

            bar.suffix = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                batch=i + 1, size=len(self.val_loader), total=bar.elapsed_td, eta=bar.eta_td,
                loss=average_loss.avg, acc=average_acc.avg
            )
            bar.next()
        bar.finish()

        is_best = False
        if average_acc.avg > self.best_acc:
            is_best = True
            self.best_acc = average_acc.avg
        return average_loss.avg, average_acc.avg, is_best

    def train(self):
        only_checkpoint_path = os.path.join(self.cfg['COMMON']['checkpoint_dir'], 'ckpts')
        if not os.path.isdir(only_checkpoint_path):
            os.makedirs(only_checkpoint_path)

        lr = self.cfg['TRAIN']['learning_rate']
        for epoch in range(self.start_epoch, self.cfg['TRAIN']['epochs'] + 1):
            lr = adjust_learning_rate(self.optimizer, epoch, lr, self.cfg['TRAIN']['schedule'],
                                      self.cfg['TRAIN']['gamma'])

            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

            loss, acc = self._train_epoch()
            val_loss, val_acc, is_best = self._evaluate()

            self.writer.add_scalar('Loss/train', loss, epoch)
            self.writer.add_scalar('Accuracy/train', acc, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            if (epoch+1) % self.cfg['COMMON']['snapshot'] == 0 or is_best:
                file_path = os.path.join(only_checkpoint_path, f'checkpoint_{epoch+1}.pth.tar')
                
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_acc': self.best_acc
                }
                if (epoch+1) % self.cfg['COMMON']['snapshot'] == 0:
                    torch.save(state, file_path)
                if is_best:
                    best_path = os.path.join(only_checkpoint_path, 'best.pth.tar')
                    if os.path.isfile(best_path):
                        os.remove(best_path)
                    torch.save(state, best_path)

        self.writer.close()
