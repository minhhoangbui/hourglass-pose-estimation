import sys
import yaml
import os
import cv2


def predict(cfg):
    estimator = Estimator(cfg)

    frame = cv2.imread(cfg['COMMON']['image_path'])
    kps = estimator.run(frame)
    for x, y in kps:
        cv2.circle(frame, center=(x, y), color=(0, 0, 255), radius=5, thickness=-1)
    cv2.imwrite(cfg['COMMON']['dest_path'], frame)


if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['COMMON']['gpu']
    from src.runner.estimator import Estimator
    predict(cfg)