from __future__ import print_function
import math
from .transform import *

__all__ = ['accuracy', 'AverageMeter', 'final_preds']


def get_preds(scores):
    """ Input: score maps in torch Tensor [batch, njoint, height, width]
        Output: coords of joint [batch, njoint, x, y]
        return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % scores.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / scores.size(3))

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
                normed_preds = preds[n, c, :].cpu().numpy() / normalize[n]
                normed_target = target[n, c, :].cpu().numpy() / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_target)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
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
    norm = np.ones((preds.shape[0], 2)) * np.array([output.size(2), output.size(3)]) / 10
    dists = calc_dists(preds, gts, norm)

    acc = np.zeros((len(idxs) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i + 1] = dist_acc(dists[idxs[i]], thr=thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc


def final_preds(output, center, scale, res):
    coords = get_preds(output)  # float type

    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < res[0] - 1 and 1 < py < res[1] - 1:
                diff = torch.Tensor([hm[py][px + 1] - hm[py][px - 1], hm[py + 1][px] - hm[py - 1][px]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds


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

