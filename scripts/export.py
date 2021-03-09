import sys
import yaml
import os
import torch
from torchsummary import summary
from src import models


def to_onnx(cfg):
    print(f"==> creating model {cfg['MODEL']['arch']}', stacks={cfg['MODEL']['num_stacks']}', "
          f"blocks={cfg['MODEL']['num_blocks']}")

    model = models.__dict__[cfg['MODEL']['arch']](num_stacks=cfg['MODEL']['num_stacks'],
                                                  num_blocks=1,
                                                  num_classes=cfg['MODEL']['num_classes'],
                                                  mobile=cfg['MODEL']['mobile'],
                                                  skip_mode=cfg['MODEL']['skip_mode'],
                                                  out_res=cfg['DATASET']['out_res'])
    summary(model, (3, cfg['MODEL']['inp_res'], cfg['MODEL']['inp_res']), device='cpu')

    print(f"=> loading checkpoint {cfg['COMMON']['checkpoint']}")
    assert os.path.isfile(cfg['COMMON']['checkpoint']), "Checkpoint doesn\'t exist"
    checkpoint = torch.load(cfg['COMMON']['checkpoint'])
    model.load_state_dict(checkpoint['state_dict'])
    dummy_input = torch.randn(1, 3, cfg['MODEL']['inp_res'], cfg['MODEL']['inp_res'])
    torch.onnx.export(model, dummy_input, cfg['COMMON']['out_onnx'], opset_version=10)


if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    to_onnx(cfg)