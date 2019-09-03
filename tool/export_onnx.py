from __future__ import print_function

import argparse
import torch
import torch.nn.parallel
import torch.optim
import os

import pose.models as models
import experiment._init_path

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main(args):

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.num_classes,
                                       mobile=args.mobile)
    model.eval()

    # optionally resume from a checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']

            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

    dummy_input = torch.randn(1, 3, args.in_res, args.in_res)
    torch.onnx.export(model, dummy_input, args.out_onnx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=16, type=int, metavar='N',
                        help='Number of keypoints')
    parser.add_argument('--out-onnx', required=True, type=str, metavar='PATH',
                        help='exported onnx file')
    parser.add_argument('--checkpoint', required=True, type=str, metavar='N',
                        help='pre-trained model checkpoint')
    parser.add_argument('--in-res', required=True, type=int, metavar='N',
                        help='input shape 128 or 256')
    parser.add_argument('--mobile', action='store_true',
                        help='Decide to use mobile architecture')
    main(parser.parse_args())
