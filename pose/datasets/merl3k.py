# ------------------------------------------------------------------------------
# Copyright (c) AWLVN
# Licensed under the MIT License.
# Written by Hoang Bui (hoang.bui@awl.com.vn)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from pycocotools.coco import COCO
from pose.datasets.common import BaseCOCO

logger = logging.getLogger(__name__)


class MERL3K(BaseCOCO):
    def __init__(self, is_train, **kwargs):
        super().__init__(is_train, **kwargs)
        self.meanstd_file = './data/merl3000/mean.pth.tar'

        if is_train:
            self.annos = COCO(os.path.join(self.json, 'train_merl.json'))
        else:
            self.annos = COCO(os.path.join(self.json, 'test_merl.json'))

        # load image file names
        self.image_set_index = self._load_image_set_index()

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]

        self.db = self._get_db()
        mean, std = self._compute_mean()
        self._get_transformation(mean, std)
        logger.info('=> load {} samples'.format(len(self.db)))

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = self.annos.imgs[index]['file_name']

        image_path = os.path.join(
            self.images, file_name)
        return image_path


def merl3k(**kwargs):
    return MERL3K(**kwargs)


merl3k.n_joints = 17
