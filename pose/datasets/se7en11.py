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
from pose.utils.imutils import load_BGR_image
import torch
from torchvision.transforms import transforms

logger = logging.getLogger(__name__)


class Se7en11(JointsDataset):
    def __init__(self, is_train, **kwargs):
        super().__init__(is_train, **kwargs)
        self.image_width = self.image_height = kwargs['inp_res']
        self.aspect_ratio = 1.0
        self.pixel_std = 200

        if self.is_train:
            self.annos = COCO(os.path.join(self.json, 'train', 'annotations.json'))
            self.images = os.path.join(kwargs['image_path'], 'train', 'images')
        else:
            self.annos = COCO(os.path.join(self.json, 'test', 'annotations.json'))
            self.images = os.path.join(kwargs['image_path'], 'test', 'images')

        cats = [cat['name']
                for cat in self.annos.loadCats(self.annos.getCatIds())]

        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))

        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.annos.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 6
        self.flip_pairs = []

        self.db = self._get_db()
        mean, std = self._compute_mean()
        mean = mean.tolist()
        std = std.tolist()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        logger.info('=> load {} samples'.format(len(self.db)))

    def _compute_mean(self):
        meanstd_file = './data/se7en11/mean.pth.tar'
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            print('==> compute mean')
            mean = torch.zeros(3)
            std = torch.zeros(3)
            cnt = 0
            for sample in self.db:
                cnt += 1
                print('{} | {}'.format(cnt, len(self.db)))
                img = load_BGR_image(sample['image'])  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.db)
            std /= len(self.db)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)

        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

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
                # obj['clean_bbox'] = [x1, y1, x2, y2]
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
            # for ipt in range(self.num_joints):
            for ipt, jpt in zip(range(self.num_joints), [4, 5, 7, 8, 9, 11]):
                joints_3d[ipt, 0] = obj['keypoints'][jpt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][jpt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][jpt * 3 + 2]
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


def se7en11(**kwargs):
    return Se7en11(**kwargs)


se7en11.n_joints = 6
