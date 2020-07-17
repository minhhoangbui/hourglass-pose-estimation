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
import numpy as np
from pycocotools.coco import COCO
from pose.datasets.common import JointsDataset

logger = logging.getLogger(__name__)


class MERL3K(JointsDataset):
    def __init__(self, is_train, **kwargs):
        super().__init__(is_train, **kwargs)
        self.image_width = kwargs['inp_res']
        self.image_height = kwargs['inp_res']
        self.aspect_ratio = 1.0
        self.pixel_std = 200
        self.meanstd_file = './data/merl3000/mean.pth.tar'

        if is_train:
            self.annos = COCO(os.path.join(self.json, 'train_merl.json'))
        else:
            self.annos = COCO(os.path.join(self.json, 'test_merl.json'))

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]

        self.db = self._get_db()
        mean, std = self._compute_mean()
        self._get_transformation(mean, std)
        logger.info('=> load {} samples'.format(len(self.db)))

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.annos.getImgIds()
        return image_ids

    def _get_db(self):
        return self._load_coco_keypoint_annotations()

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.annos.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.annos.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.annos.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = self.annos.imgs[index]['file_name']

        image_path = os.path.join(
            self.images, file_name)
        return image_path


def merl3k(**kwargs):
    return MERL3K(**kwargs)


merl3k.n_joints = 17
