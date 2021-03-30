import sys
import yaml
import os


def train(cfg):
    cfg['COMMON']['checkpoint_dir'] = os.path.join(cfg['COMMON']['checkpoint_dir'],
                                                   '{}_{}_s{}_{}_{}'.format(cfg['DATASET']['name'], cfg['MODEL']['arch'],
                                                                            cfg['MODEL']['num_stacks'],
                                                                            'mobile' if cfg['MODEL']['mobile'] else
                                                                            'non-mobile',
                                                                            'all' if cfg['MODEL']['subset'] is None else
                                                                            cfg['MODEL']['subset']))
    if not os.path.isdir(cfg['COMMON']['checkpoint_dir']):
        os.makedirs(cfg['COMMON']['checkpoint_dir'])

    n_joints = datasets.__dict__[cfg['DATASET']['name']].n_joints if cfg['MODEL']['subset'] is None else \
        len(cfg['MODEL']['subset'])

    trainer = Trainer(cfg, n_joints)
    trainer.train()


def val(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    n_joints = datasets.__dict__[cfg['DATASET']['name']].n_joints if cfg['MODEL']['subset'] is None else \
        len(cfg['MODEL']['subset'])

    print(f"==> creating model '{cfg['MODEL']['arch']}', stacks={cfg['MODEL']['num_stacks']}")
    model = models.__dict__[cfg['MODEL']['arch']](num_stacks=cfg['MODEL']['num_stacks'],
                                                  num_blocks=1,
                                                  num_classes=n_joints,
                                                  mobile=cfg['MODEL']['mobile'],
                                                  skip_mode=cfg['MODEL']['skip_mode'],
                                                  out_res=cfg['DATASET']['out_res'])
    summary(model, (3, cfg['DATASET']['inp_res'], cfg['DATASET']['inp_res']), device='cpu')
    model = torch.nn.DataParallel(model).to(device)
    if os.path.isfile(cfg['COMMON']['resume']):
        from src.runner.evaluator import Evaluator
        checkpoint = torch.load(cfg['COMMON']['resume'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model {cfg['COMMON']['resume']}")
        evaluator = Evaluator(device, cfg)
        _, _ = evaluator.evaluate(model)


if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['COMMON']['gpu']
    import torch
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    from torchsummary import summary
    from src import datasets, models
    from src.runner.trainer import Trainer

    if cfg['COMMON']['evaluate_only']:
        val(cfg)
    else:
        train(cfg)