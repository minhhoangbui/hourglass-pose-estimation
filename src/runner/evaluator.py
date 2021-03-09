import torch
import torch.utils.data
from progress.bar import Bar

from src.loss.mse import MSELoss
from src import datasets
from src.utils.evaluation import AverageMeter, accuracy


class Evaluator(object):
    def __init__(self, device, cfg):
        self.cfg = cfg
        self.device = device
        self.criterion = MSELoss(use_target_weight=True)

        val_dataset = datasets.__dict__[cfg['DATASET']['name']](is_train=False, **cfg['DATASET'])
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg['TRAIN']['val_batch'], shuffle=True,
            num_workers=cfg['TRAIN']['num_workers'], pin_memory=True
        )

        if self.cfg['MODEL']['subset']:
            self.idxs = torch.LongTensor(cfg['MODEL']['subset'])

    def evaluate(self, model):
        model.eval()
        average_loss = AverageMeter()
        average_acc = AverageMeter()

        bar = Bar('Evaluating', max=len(self.val_loader))
        idxs = self.cfg['MODEL']['subset']

        for i, (images, heatmaps, meta) in enumerate(self.val_loader):
            if idxs:
                heatmaps = torch.index_select(heatmaps, 1, self.idxs)
            images = images.to(self.device)
            heatmaps = heatmaps.to(self.device, non_blocking=True)
            target_weight = meta['target_weight'].to(self.device, non_blocking=True)

            outputs = model(images)
            last_hms = outputs[-1]
            loss = self.criterion(outputs, heatmaps, target_weight)

            acc = accuracy(last_hms, heatmaps, idxs, thr=self.cfg['COMMON']['pck'])

            average_loss.update(loss.item(), images.size(0))
            average_acc.update(acc[0], images.size(0))

            bar.suffix = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                batch=i + 1, size=len(self.val_loader), total=bar.elapsed_td, eta=bar.eta_td,
                loss=average_loss.avg, acc=average_acc.avg
            )

            bar.next()
        bar.finish()
        return average_loss.avg, average_acc.avg

