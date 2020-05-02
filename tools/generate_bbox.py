import os
import cv2
from pascal_voc_writer import Writer
import sys
sys.path.append('../')
from detector import Detector
import glob
import shutil


def process(src, images_dest, annos_dest):
    detector = Detector(graph='/home/hoang/fare-evasion-detection/tests/models/personface_frcnn_resnet50.pb',
                        conf=0.6)
    files = glob.glob(src + '*.jpg')
    for file in files:
        print(file)
        name = os.path.basename(file)
        _name, ext = os.path.splitext(name)
        frame = cv2.imread(file)
        w, h = frame.shape[:2]
        writer = Writer(file, w, h)
        person_boxes, _ = detector.process_frame(frame)
        if len(person_boxes) == 0:
            continue
        shutil.move(file, os.path.join(images_dest, _name + '.jpg'))
        for box in person_boxes:
            writer.addObject('person', box[1], box[0], box[3], box[2])
            # cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 1)
        writer.save(os.path.join(annos_dest, _name + '.xml'))
        # cv2.imwrite(os.path.join(dest, name), frame)


if __name__ == '__main__':
    origin = '05/AVC9003_20190830_050000/'
    src = '/mnt/hdd3tb/project4/JR_Dataset/Dataset/cam3/images/' + origin
    dest = '/mnt/hdd3tb/project4/JR_Dataset/Dataset/cam3/processed/' + origin
    images_dest = os.path.join(dest, 'images')
    annos_dest = os.path.join(dest, 'annos')
    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(annos_dest, exist_ok=True)
    process(src, images_dest, annos_dest)
