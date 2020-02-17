import json
import cv2
import os
import time


def draw_bbox_keypoints(images_dir, annotation_file):
    with open(annotation_file, 'r') as fp:
        annotations = json.load(fp)
    ids_kps = dict()
    for anno in annotations['annotations']:
        if anno['image_id'] not in ids_kps:
            ids_kps[anno['image_id']] = [(anno['bbox'], anno['keypoints'])]
        else:
            print(anno['image_id'])
            ids_kps[anno['image_id']].append((anno['bbox'], anno['keypoints']))

    for image in annotations['images']:
        image_file = os.path.join(images_dir, image['file_name'])
        if os.path.isfile(image_file):
            frame = cv2.imread(image_file)
        else:
            raise FileNotFoundError(image_file)
        for (bbox, kps) in ids_kps[image['id']]:
            _x1, _y1, _x2, _y2 = bbox
            cv2.rectangle(frame, (_x1, _y1), (_x2, _y2), (0, 255, 0))
            for i in range(25):
                if kps[i * 3 + 2] == 0:
                    pass
                else:
                    x = kps[i * 3]
                    y = kps[i * 3 + 1]
                    cv2.circle(frame, center=(x, y), color=(255, 0, 0), radius=5, thickness=-1)
            cv2.imshow("Results", frame)
            # time.sleep(1)
            keypress = cv2.waitKey(25)
            if keypress & 0xFF == ord(' '):
                while True:
                    keypress = cv2.waitKey(1)
                    if keypress & 0xFF == ord(' '):
                        break


if __name__ == '__main__':
    images_dir = '/Users/minhhoang/models/datasets/images'
    annotation_file = '/Users/minhhoang/models/datasets/annotations.json'
    draw_bbox_keypoints(images_dir, annotation_file)