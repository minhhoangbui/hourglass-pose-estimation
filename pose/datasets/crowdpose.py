import logging
import os

import numpy as np
from crowdposetools.coco import COCO
from pose.datasets.common import JointsDataset


logger = logging.getLogger(__name__)


class CrowdPose(JointsDataset):
    """
    kps_names =['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck']
    kps_lines = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (12, 13)]
    """
    def __init__(self, is_train, **kwargs):
        super().__init__(is_train, **kwargs)
        self.image_width = kwargs['inp_res']
        self.image_height = kwargs['inp_res']
        self.aspect_ratio = 1.0
        self.meanstd_file = './data/crowdpose/mean.pth.tar'
        if self.is_train:
            self.coco = COCO(os.path.join(self.json, 'crowdpose_train.json'))
        else:
            self.coco = COCO(os.path.join(self.json, 'crowdpose_test.json'))

        self.num_joints = 14
        self.flip_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]

        self.db = self._get_db()
        mean, std = self._compute_mean()
        self._get_transformation(mean, std)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _load_data(self):
        gt_db = []
        for aid in self.coco.anns.keys():
            ann = self.coco.anns[aid]

            x, y, w, h = ann['bbox']
            img = self.coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if x2 >= x1 and y2 >= y1:  # ann['area'] > 0 #tobi: CrowdPose does not provide area
                bbox = [x1, y1, x2 - x1, y2 - y1]
            else:
                continue
            x, y, w, h = bbox
            center = np.array([x + w * 0.5, y + h * 0.5])
            if w > self.aspect_ratio * h:
                h = w / self.aspect_ratio
            elif w < self.aspect_ratio * h:
                w = h * self.aspect_ratio
            scale = np.array([w, h]) * 1.25

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = ann['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = ann['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = ann['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0
            gt_db.append({
                'image': os.path.join(self.images, self.coco.imgs[ann['image_id']]['file_name']),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'imgnum': 0
            })
        return gt_db


def crowdpose(**kwargs):
    return CrowdPose(**kwargs)


crowdpose.n_joints = 14
