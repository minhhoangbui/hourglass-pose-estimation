from __future__ import print_function

import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
import glob
import torch
import torch.nn.parallel
import torch.optim
import cv2
from collections import OrderedDict
import experiment._init_path
from pose.models.hourglass import Bottleneck, HourglassNet
import numpy as np
import time


class PosePredictor:
    def __init__(self, args):
        self.model = HourglassNet(Bottleneck, num_stacks=args['num_stacks'], num_blocks=args['num_blocks'],
                                  num_classes=args['num_classes'], mobile=args['mobile'])
        self.bbox = args['bbox']
        self.dataset = args['dataset']
        self.input_size = args['input_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args['checkpoint'], map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = 0.02
        print("finish loading models in %s" % self.device)

    def preprocess_bbox(self, bbox):
        in_frame = bbox / 255.0
        if 'coco' in self.dataset:
            in_frame = (in_frame - np.array([[[0.4003, 0.4314, 0.4534]]])) / np.array([[[0.2466, 0.2467, 0.2562]]])
        elif 'mpii' in self.dataset:
            in_frame = (in_frame - np.array([[[0.4327, 0.4440, 0.4404]]])) / np.array([[[0.2468, 0.2410, 0.2458]]])
        elif 'merl' in self.dataset:
            in_frame = (in_frame - np.array([[[0.4785, 0.5036, 0.5078]]])) / np.array([[[0.2306, 0.2289, 0.2326]]])
        elif 'se7en11' in self.dataset:
            in_frame = (in_frame - np.array([[[0.5109, 0.5502, 0.5285]]])) / np.array([[[0.2772, 0.2416, 0.2478]]])

        in_frame = cv2.resize(in_frame, self.input_size)
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((1, 3, self.input_size[0], self.input_size[1]))
        in_frame = torch.from_numpy(in_frame).float().to(self.device)
        return in_frame

    def post_process_heatmap(self, heatmap):
        if self.device == torch.device('cuda'):
            heatmap = heatmap.cpu().numpy()
        kplst = []
        for i in range(heatmap.shape[0]):
            _map = heatmap[i, :, :]
            ind = np.unravel_index(np.argmax(_map), _map.shape)
            if _map[ind] > self.threshold:
                kplst.append((int(ind[1]), int(ind[0]), _map[ind]))
            else:
                kplst.append((0, 0, 0))
        kplst = np.array(kplst)
        return kplst

    def run(self, frame):
        in_frame = self.preprocess_bbox(frame)
        start = time.time()
        if self.device == 'cpu':
            heatmap = self.model(in_frame)[-1].cpu().detach().numpy()[0]
        else:
            heatmap = self.model(in_frame)[-1].detach()[0]
        end = time.time()
        print("Inference time on %s: %0.3f" % (self.device, end - start))
        kps = self.post_process_heatmap(heatmap)

        scale_x = frame.shape[1] * 1.0 / self.input_size[0]
        scale_y = frame.shape[0] * 1.0 / self.input_size[1]
        kps = [kps[:, 0] * scale_x * 4, kps[:, 1] * scale_y * 4]

        kps = np.asarray(kps, dtype=np.float16).transpose()
        return kps


if __name__ == '__main__':
    args = {
        'checkpoint': '/Users/minhhoang/models/se7en11_s2_b1_mobile_all/checkpoint.pth.tar',
        # 'checkpoint': '/mnt/ssd2/Users/hoangbm/checkpoint/pose-estimation/se7en11_s2_b1_mobile_all/model_best.pth.tar',
        'num_stacks': 2,
        'num_blocks': 1,
        'num_classes': 6,
        'mobile': True,
        'input_size': (256, 256),
        'dataset': 'se7en11',
        'bbox': [500, 0, 955, 690],
        'video': '/Users/minhhoang/Downloads/video.mp4',
        'image': '/Users/minhhoang/pose-distillation-pytorch/data/se7en11/color1581647146889_0002.jpg',
        'folder': '/Users/minhhoang/models/datasets/train/images'
    }

    x_min, y_min, x_max, y_max = args['bbox']

    predictor = PosePredictor(args)
    image_list = glob.glob(args['folder'] + '/*.jpg')
    img = args['image']

    # image
    # frame = cv2.imread(img)
    # cropped_frame = frame[y_min: y_max, x_min: x_max]
    # start = time.time()
    # kps = predictor.run(cropped_frame)
    # end = time.time()
    # kps[:, 0] += x_min
    # kps[:, 1] += y_min
    # for x, y in kps:
    #     cv2.circle(frame, center=(x, y), color=(0, 0, 255), radius=5, thickness=-1)
    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    # cv2.imshow("Results", frame)
    # cv2.waitKey(0)
    # print("Total inference time on %s: %0.3f" % (predictor.device, end - start))

    # video
    if not os.path.isfile(args['video']):
        raise FileNotFoundError()

    reader = cv2.VideoCapture(args['video'])

    if not reader.isOpened():
        raise OSError('Cannot open video')
    _, frame = reader.read()

    while True:
        _, frame = reader.read()
        if frame is None:
            break
        cropped_frame = frame[y_min: y_max, x_min: x_max]
        kps = predictor.run(cropped_frame)
        kps[:, 0] += x_min
        kps[:, 1] += y_min
        for x, y in kps:
            cv2.circle(frame, center=(x, y), color=(0, 0, 255), radius=5, thickness=-1)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        cv2.imshow("Results", frame)
        keypress = cv2.waitKey(25)
        if keypress & 0xFF == ord('q'):
            break

        if keypress & 0xFF == ord(' '):
            while True:
                keypress = cv2.waitKey(1)
                if keypress & 0xFF == ord(' '):
                    break
    cv2.destroyAllWindows()
    reader.release()
