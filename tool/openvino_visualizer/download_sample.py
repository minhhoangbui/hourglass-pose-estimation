import json
import os
import argparse
import shutil


def main(args):
    with open(args.json) as f:
        collections = json.load(f)
        collections = collections[:1000]
        if args.type == 'coco':
            for coll in collections:
                if coll["isValidation"]:
                    image_path = os.path.join(args.image, 'val2017')
                else:
                    image_path = os.path.join(args.image, 'train2017')
                image_path = os.path.join(image_path, coll['img_paths'])
                shutil.copyfile(image_path, '/mnt/hdd3tb/Users/hoang/{}/{}'.format(args.type, coll['img_paths']))
        elif args.type == 'mpii':
            for coll in collections:
                image_path = os.path.join(args.image, coll['img_paths'])
                shutil.copyfile(image_path, '/mnt/hdd3tb/Users/hoang/{}/{}'.format(args.type, coll['img_paths']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--type', type=str)
    args = parser.parse_args()
    main(args)