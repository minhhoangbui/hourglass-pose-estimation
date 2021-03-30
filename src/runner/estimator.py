import torch
import os
import time
import numpy as np
import cv2
from collections import OrderedDict
from src import models
from src.utils.inference import get_final_preds_v1, get_final_preds_v2


class Estimator:
    def __init__(self, cfg):
        print(f"==> creating model '{cfg['MODEL']['arch']}', stacks={cfg['MODEL']['num_stacks']}")

        self.model = models.__dict__[cfg['MODEL']['arch']](num_stacks=cfg['MODEL']['num_stacks'],
                                                           num_blocks=1,
                                                           num_classes=cfg['MODEL']['num_classes'],
                                                           mobile=cfg['MODEL']['mobile'],
                                                           skip_mode=cfg['MODEL']['skip_mode'],
                                                           out_res=cfg['COMMON']['out_res'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = cfg['COMMON']['dataset']
        self.input_size = (cfg['COMMON']['in_res'], cfg['COMMON']['in_res'])
        self.threshold = 0.02

        if os.path.isfile(cfg['COMMON']['resume']):
            checkpoint = torch.load(cfg['COMMON']['resume'], map_location=self.device)
            loaded_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                loaded_dict[k[7:]] = v
            self.model.load_state_dict(loaded_dict)

            print(f"Loaded model {cfg['COMMON']['resume']}")
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

    def post_process_heatmap_v1(self, heatmaps, output_size):
        if self.device == torch.device('cuda'):
            heatmaps = heatmaps.cpu().numpy()[0]
        else:
            heatmaps = heatmaps.numpy()[0]
        kplst = []
        for i in range(heatmaps.shape[0]):
            _map = heatmaps[i, :, :]
            ind = np.unravel_index(np.argmax(_map), _map.shape)
            if _map[ind] > self.threshold:
                kplst.append((int(ind[1]), int(ind[0]), _map[ind]))
            else:
                kplst.append((0, 0, 0))
        kplst = np.array(kplst)
        scale_x = output_size[0] * 1.0 / self.input_size[0]
        scale_y = output_size[1] * 1.0 / self.input_size[1]
        kps = [kplst[:, 0] * scale_x * 4, kplst[:, 1] * scale_y * 4]
        kps = np.asarray(kps, dtype=np.int).transpose()
        return kps

    @staticmethod
    def post_process_heatmap_v2(heatmap, output_size):
        center = np.array([round(output_size[0] * 0.5), round(output_size[1] * 0.5)])
        scale = np.array([output_size[0] * 4.0 / 200 / heatmap.shape[2],
                          output_size[1] * 4.0 / 200 / heatmap.shape[3]])
        kps = get_final_preds_v1(heatmap, center, scale, output_size)
        return kps.astype(np.int)

    def run(self, frame):
        in_frame = self.preprocess_bbox(frame)

        start = time.time()
        heatmaps = self.model(in_frame)[-1].detach()
        end = time.time()
        print(f"Inference time on {self.device}: %0.3f" % (end - start))

        kps = self.post_process_heatmap_v2(heatmaps, (frame.shape[1], frame.shape[0]))
        return kps
