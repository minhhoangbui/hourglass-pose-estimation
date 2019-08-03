import copy
import math
import os
import pickle

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

BODY_PARTS_KPT_IDS = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
                      [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]


class ConvertKeypoints(object):
    def __call__(self, sample):
        label = sample['label']
        h, w, _ = sample['image'].shape
        keypoints = label['keypoints']
        for keypoint in keypoints:  # keypoint[2] == 0: occluded, == 1: visible, == 2: not in image
            if keypoint[0] == keypoint[1] == 0:
                keypoint[2] = 2
            if (keypoint[0] < 0
                    or keypoint[0] >= w
                    or keypoint[1] < 0
                    or keypoint[1] >= h):
                keypoint[2] = 2
        for other_label in label['processed_other_annotations']:
            keypoints = other_label['keypoints']
            for keypoint in keypoints:
                if keypoint[0] == keypoint[1] == 0:
                    keypoint[2] = 2
                if (keypoint[0] < 0
                        or keypoint[0] >= w
                        or keypoint[1] < 0
                        or keypoint[1] >= h):
                    keypoint[2] = 2
        label['keypoints'] = self._convert(label['keypoints'], w, h)

        for other_label in label['processed_other_annotations']:
            other_label['keypoints'] = self._convert(other_label['keypoints'], w, h)
        return sample

    def _convert(self, keypoints, w, h):
        # Nose, Neck, R hand, L hand, R leg, L leg, Eyes, Ears
        reorder_map = [1, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        converted_keypoints = list(keypoints[i - 1] for i in reorder_map)
        converted_keypoints.insert(1, [(keypoints[5][0] + keypoints[6][0]) / 2,
                                       (keypoints[5][1] + keypoints[6][1]) / 2, 0])  # Add neck as a mean of shoulders
        if keypoints[5][2] == 2 and keypoints[6][2] == 2:
            converted_keypoints[1][2] = 2
        elif keypoints[5][2] == 3 and keypoints[6][2] == 3:
            converted_keypoints[1][2] = 3
        elif keypoints[5][2] == 1 and keypoints[6][2] == 1:
            converted_keypoints[1][2] = 1
        if (converted_keypoints[1][0] < 0
                or converted_keypoints[1][0] >= w
                or converted_keypoints[1][1] < 0
                or converted_keypoints[1][1] >= h):
            converted_keypoints[1][2] = 2
        return converted_keypoints


class MERL3kDataset(Dataset):
    def __init__(self, is_train=True, **kwargs):
        self._images_folder = kwargs['image_path']
        self.is_train = is_train
        self._sigma = kwargs['sigma']
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self._transform = ConvertKeypoints()
        if is_train:
            with open(os.path.join(kwargs['anno_path'], 'train_annotation.pkl'), 'rb') as f:
                self._labels = pickle.load(f)
        else:
            with open(os.path.join(kwargs['anno_path'], 'test_annotation.pkl'), 'rb') as f:
                self._labels = pickle.load(f)

    def __getitem__(self, idx):
        label = copy.deepcopy(self._labels[idx])  # label modified in transform
        image = cv2.imread(os.path.join(self._images_folder, label['img_paths']), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.inp_res, self.inp_res), interpolation=cv2.INTER_AREA)
        sample = {
            'label': label,
            'image': image
        }

        if self._transform:
            sample = self._transform(sample)
        self._stride = self.inp_res // self.out_res

        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps

        image = sample['image'].astype(np.float32)
        image = (image - 128) / 256
        sample['image'] = image.transpose((2, 0, 1))
        meta = {'index': idx, 'center': label['objpos'], 'scale': label['scale_provided']}
        return sample['image'], sample['keypoint_maps'], meta

    def __len__(self):
        return len(self._labels)

    def _generate_keypoint_maps(self, sample):
        n_keypoints = 18
        keypoint_maps = np.zeros(shape=(n_keypoints,
                                        self.out_res, self.out_res), dtype=np.float32)

        label = sample['label']
        for keypoint_idx in range(n_keypoints):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
            for another_annotation in label['processed_other_annotations']:
                keypoint = another_annotation['keypoints'][keypoint_idx]
                if keypoint[2] <= 1:
                    self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
        return keypoint_maps

    @staticmethod
    def _add_gaussian(keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                    (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1


def merl3k(**kwargs):
    return MERL3kDataset(**kwargs)


merl3k.njoints = 18

