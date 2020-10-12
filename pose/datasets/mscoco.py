# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from pycocotools.coco import COCO
from pose.datasets.common import BaseCOCO

logger = logging.getLogger(__name__)


class MSCOCO(BaseCOCO):
    """
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },

    "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    """
    def __init__(self, is_train, **kwargs):
        super().__init__(is_train, **kwargs)
        if is_train:
            self.image_set = 'train2017'
        else:
            self.image_set = 'val2017'
        self.annos = COCO(self._get_ann_file_keypoint())

        # load image file names
        self.image_set_index = self._load_image_set_index()

        self.num_joints = 17
        self.meanstd_file = './data/coco/mean.pth.tar'
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]

        self.db = self._get_db()
        mean, std = self._compute_mean()
        self.transform = BaseCOCO._get_transformation(mean, std)

        # if is_train and cfg.DATASET.SELECT_DATA:
        #     self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.json,
                            prefix + '_' + self.image_set + '.json')

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        image_path = os.path.join(
            self.images, prefix, file_name)

        return image_path


def mscoco(**kwargs):
    return MSCOCO(**kwargs)


mscoco.n_joints = 17
