import logging
import os
from pycocotools.coco import COCO
from pose.datasets.common import BaseCOCO

logger = logging.getLogger(__name__)


class Hands(BaseCOCO):
    def __init__(self, is_train, **kwargs):
        super().__init__(is_train, **kwargs)
        self.meanstd_file = './data/hands/mean.pth.tar'

        if is_train:
            self.images_set = 'train2017'
            self.annos = COCO(os.path.join(self.json, 'hands_train.json'))
        else:
            self.images_set = 'val2017'
            self.annos = COCO(os.path.join(self.json, 'hands_val.json'))

        self.image_set_index = self._load_image_set_index()

        self.num_joints = 22
        self.flip_pairs = [[i, i + 11] for i in range(11)]

        self.db = self._get_db()
        mean, std = self._compute_mean()
        self.transform = self._get_transformation(mean, std)

        logger.info('=> load {} samples'.format(len(self.db)))

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = self.annos.imgs[index]['file_name']

        image_path = os.path.join(
            self.images, self.images_set, file_name)
        return image_path


def hands(**kwargs):
    return Hands(**kwargs)


hands.n_joints = 22
