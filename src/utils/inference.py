import numpy as np
import cv2
import math
import torch
from src.utils.evaluation import get_preds
from src.utils.transforms import transform_preds


def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            inv_hessian = np.linalg.inv(hessian)
            # offset = - inv_hessian * derivative
            offset = - np.dot(inv_hessian, derivative)
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def gaussian_blur(hm, kernel=11):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i, j] = dr[border: -border, border: -border].copy()
            hm[i, j] *= origin_max / np.max(hm[i, j])
    return hm


def get_final_preds_v1(hms, center, scale, output_size):
    coords = get_preds(hms)[0]  # float type
    heatmap_height = hms.shape[2]
    heatmap_width = hms.shape[3]

    # pose-processing
    for p in range(coords.size(0)):
        hm = hms[0][p]
        px = int(math.floor(coords[p][0] + 0.5))
        py = int(math.floor(coords[p][1] + 0.5))
        if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
            diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2],
                                 hm[py][px - 1] - hm[py - 2][px - 1]])
            coords[p] += diff.sign() * .25
    # coords += 0.5

    # Transform back
    preds = transform_preds(coords, center, scale, output_size)

    return preds


def get_final_preds_v2(hms, center, scale, output_size):
    coords = get_preds(hms)[0]
    hms = hms.numpy()

    # post-processing
    hms = gaussian_blur(hms)
    hms = np.maximum(hms, 1e-10)
    hms = np.log(hms)

    for p in range(coords.shape[1]):
        coords[p] = taylor(hms[0][p], coords[p])

    # Transform back

    preds = transform_preds(
        coords, center, scale, output_size
    )
    return preds


if __name__ == '__main__':
    hms = torch.randn((4, 17, 64, 64))
    center = [[32, 32] for _ in range(4)]
    scale = [[0.5, 0.5] for _ in range(4)]
    output_size = [256, 256]
    preds = get_final_preds_v1(hms, center, scale)
    for pred in preds:
        print(pred.size())