import torch
import os
import time
import numpy as np
import cv2
from src import models


class Estimator:
    def __init__(self, cfg):
        print(f"==> creating model '{cfg['MODEL']['arch']}', stacks={cfg['MODEL']['num_stacks']}")

        self.model = models.__dict__[cfg['MODEL']['arch']](num_stacks=cfg['MODEL']['num_stacks'],
                                                      num_blocks=1,
                                                      num_classes=cfg['MODEL']['num_classes'],
                                                      mobile=cfg['MODEL']['mobile'],
                                                      skip_mode=cfg['MODEL']['skip_mode'],
                                                      out_res=cfg['DATASET']['out_res'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = cfg['COMMON']['dataset']
        self.input_size = cfg['COMMON']['input_size']
        self.threshold = 0.02

        if os.path.isfile(cfg['COMMON']['resume']):
            checkpoint = torch.load(cfg['COMMON']['resume'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError('Checkpoint not found')

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