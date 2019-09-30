import logging
import os
from torchvision.transforms import transforms
import torch
import numpy as np
from crowdposetools.coco import COCO
from pose.datasets.common import JointsDataset
from pose.utils.imutils import load_BGR_image

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
        if self.is_train:
            self.coco = COCO(os.path.join(self.json, 'crowdpose_train.json'))
        else:
            self.coco = COCO(os.path.join(self.json, 'crowdpose_test.json'))

        self.num_joints = 14
        self.flip_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        self.db = self._load_data()
        mean, std = self._compute_mean()
        mean = mean.tolist()
        std = std.tolist()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        logger.info('=> load {} samples'.format(len(self.db)))

    def _compute_mean(self):
        meanstd_file = './data/crowdpose/mean.pth.tar'
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
