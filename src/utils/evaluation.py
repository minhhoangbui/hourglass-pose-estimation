from __future__ import print_function
import torch
import numpy as np

__all__ = ['accuracy', 'AverageMeter']


def get_preds(batch_heatmaps):
    """ Input: batch_heatmaps in torch Tensor [batch, njoint, height, width]
        Output: coords of joint [batch, njoint, 2]
        return type: torch.LongTensor
    """
    assert batch_heatmaps.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(batch_heatmaps.view(batch_heatmaps.size(0),
                                                batch_heatmaps.size(1), -1), 2)

    maxval = maxval.view(batch_heatmaps.size(0), batch_heatmaps.size(1), 1)
    idx = idx.view(batch_heatmaps.size(0), batch_heatmaps.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % batch_heatmaps.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / batch_heatmaps.size(3)) + 1

    pred_mask = maxval.gt(0.).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = np.zeros((preds.size(1), preds.size(0)))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n, c, :], target[n, c, :]) / normalize[n]
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist = dists[dists != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1


def accuracy(output, target, idxs=None, thr=0.5):
    """
    Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    """
    if idxs is None:
        idxs = list(range(output.shape[1]))
    preds = get_preds(output)
    gts = get_preds(target)
    norm = torch.ones(preds.size(0)) * output.size(3) / 10
    dists = calc_dists(preds, gts, norm)

    acc = np.zeros((len(idxs) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i + 1] = dist_acc(dists[i], thr=thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    heatmaps = torch.rand((4, 17, 64, 64))
    coords = get_preds(heatmaps)
    print(coords.size())