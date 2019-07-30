from __future__ import print_function
from scipy.io import loadmat
from scipy import transpose
import numpy as np
import argparse


def main(args):

    SC_BIAS = 0.6
    dict_ = loadmat('tool/data/detections_our_format.mat')
    dataset_joints = dict_['dataset_joints']
    joints_missing = dict_['jnt_missing']
    pos_gt_src = dict_['pos_gt_src']
    headboxes_src = dict_['headboxes_src']

    preds = loadmat(args.preds)['preds']
    pos_pred_src = transpose(preds, [1, 2, 0])

    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]

    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]

    joints_visible = 1 - joints_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, joints_visible)
    jnt_count = np.sum(joints_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < args.threshold), joints_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    # save
    rng = np.arange(0, 0.5, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err < threshold, joints_visible)
        pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    print('\nPrediction file: {}\n'.format(args.preds))
    print("Head,   Shoulder, Elbow,  Wrist,   Hip ,     Knee  , Ankle ,  Mean")
    print('{:.2f}  {:.2f}     {:.2f}  {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(PCKh[head],
                                                                                         0.5 * (PCKh[lsho] + PCKh[rsho]) \
                                                                                         , 0.5 * (PCKh[lelb] + PCKh[
            relb]), 0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]), 0.5 * (PCKh[lkne] + PCKh[rkne]) \
                                                                                         , 0.5 * (PCKh[lank] + PCKh[
            rank]), np.mean(PCKh)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPII PCKh Evaluation')
    parser.add_argument('--preds', default='checkpoint/mpii/hg_s2_b1/preds.mat',
                        type=str, metavar='PATH',
                        help='path to result (default: checkpoint/mpii/hg_s2_b1/preds.mat)')
    parser.add_argument('--threshold', type=int, default=0.5,
                        help='Threshold in PCKh metrics')
    args = parser.parse_args()
    main(args)
