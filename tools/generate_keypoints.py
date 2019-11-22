from __future__ import print_function

import argparse
import torch
import json
import os
import numpy as np
import pose.models as models
import experiment._init_path
from torchvision.transforms import transforms
import cv2

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def main(args):
    # Load model
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.num_classes,
                                       mobile=args.mobile)
    model = torch.nn.DataParallel(model).cuda()
    print('    Total params of teacher model: %.2fM'
          % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    meanstd_file = './data/coco_v2/mean.pth.tar'
    meanstd = torch.load(meanstd_file)
    transformations = transforms.Normalize(mean=meanstd['mean'], std=meanstd['std'])

    # image = cv2.imread('tools/image.jpg', cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    # cropped_image = cv2.resize(image, (args.in_res, args.in_res))
    # cropped_image = cropped_image.transpose((2, 0, 1))
    # cropped_image = torch.tensor(cropped_image).float() / 255.0
    # cropped_image = transformations(cropped_image)
    # cropped_image = cropped_image.unsqueeze_(0)
    # output = model(cropped_image)
    # output = output[-1].detach().cpu()
    # output = output.numpy()
    # output = output[0]
    # print(output.shape)
    # left_wrist = output[0]
    # right_wrist = output[1]
    # scale_x = image.shape[1] * 1.0 / args.in_res
    # scale_y = image.shape[0] * 1.0 / args.in_res
    # left_ind = np.unravel_index(np.argmax(left_wrist), left_wrist.shape)
    # left_ind = [left_ind[1] * scale_x * 4, left_ind[0] * scale_y * 4]
    # print(left_ind)
    # right_ind = np.unravel_index(np.argmax(right_wrist), right_wrist.shape)
    # right_ind = [right_ind[1] * scale_x * 4, right_ind[0] * scale_y * 4]
    # print(right_ind)
    # cv2.circle(image, center=(int(left_ind[0]), int(left_ind[1])), color=(255, 0, 0), radius=10)
    # cv2.circle(image, center=(int(right_ind[0]), int(right_ind[1])), color=(255, 0, 0), radius=10)
    # cv2.imwrite('tools/image_1.jpg', image)

    with open(args.json, 'r') as fp:
        jfile = json.load(fp)
        for anno in jfile['annotations']:
            image_id = anno['image_id']
            x, y, w, h = anno['bbox']
            fname = None
            for image in jfile['images']:
                if image['id'] == image_id:
                    fname = image['file_name']
            if fname is None:
                raise ValueError("No picture found")
            image_path = os.path.join(args.images, fname)
            if not os.path.isfile(image_path):
                raise ValueError("No picture available at %s" %image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            image = image[int(y):int(y) + int(h), int(x): int(x) + int(w), :]
            image = cv2.resize(image, (args.in_res, args.in_res))
            image = image.transpose((2, 0, 1))
            image = torch.tensor(image).float() / 255.0
            image = transformations(image)
            image = image.unsqueeze_(0)
            output = model(image)
            output = output[-1].detach().cpu()
            output = output.numpy()
            output = output[0]
            left_wrist = output[0]
            right_wrist = output[1]
            scale_x = h * 1.0 / args.in_res
            scale_y = w * 1.0 / args.in_res
            left_ind = np.unravel_index(np.argmax(left_wrist), left_wrist.shape)
            left_ind = [left_ind[1] * scale_x * 4, left_ind[0] * scale_y * 4]
            right_ind = np.unravel_index(np.argmax(right_wrist), right_wrist.shape)
            right_ind = [right_ind[1] * scale_x * 4, right_ind[0] * scale_y * 4]
            anno['keypoints'][21] = int(left_ind[0] + x)
            anno['keypoints'][22] = int(left_ind[1] + y)
            anno['keypoints'][23] = 2

            anno['keypoints'][24] = int(right_ind[0] + x)
            anno['keypoints'][25] = int(right_ind[1] + y)
            anno['keypoints'][26] = 2

        # for image in jfile['images']:
        #     fulldir, fname = os.path.split(image['url'])
        #     url, _ = os.path.split(fulldir)
        #     image['url'] = os.path.join(url, fname)

    basedir = os.path.dirname(args.json)
    with open(os.path.join(basedir, 'wrists4.json'), 'w') as fp:
        json.dump(jfile, fp, default=convert)


# def main_2(json_file=None):
#     base, _ = os.path.split(json_file)
#     with open(json_file, 'r') as fp:
#         data = json.load(fp)
#     for image in data['images']:
#         fulldir, fname = os.path.split(image['url'])
#         url, _ = os.path.split(fulldir)
#         image['url'] = os.path.join(url, fname)
#     with open(os.path.join(base, 'processed_wrist4.json'), 'w') as fp:
#         json.dump(data, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=2, type=int, metavar='N',
                        help='Number of keypoints')
    parser.add_argument('--checkpoint', required=True, type=str, metavar='N',
                        help='pre-trained model checkpoint')
    parser.add_argument('--in-res', required=True, type=int, metavar='N',
                        help='input shape 128 or 256')
    parser.add_argument('--mobile', action='store_true',
                        help='Decide to use mobile architecture')
    parser.add_argument('--images', type=str)
    parser.add_argument('--json', type=str)
    args = parser.parse_args()
    main(args)

