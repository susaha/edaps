# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmseg.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from tools.parser_argument_help_str import paht1, paht2, wm1, wm2
import sys
from mmdet.models.builder import build_train_model
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume', action='store_true', help='resume from the latest checkpoint automatically')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='(Deprecated, please use --gpu-id) number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='(Deprecated, please use --gpu-id) ids of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--diff-seed', action='store_true', help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options', nargs='+', action=DictAction, help=paht1)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help=paht2)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--auto-scale-lr', action='store_true', help='enable automatically scaling LR.')
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError('--options and --cfg-options cannot be both specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options
    return args


def main(args):
    args = parse_args(args)
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.train_cfg.work_dir = cfg.work_dir
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support single GPU mode in non-distributed training. Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(wm2)
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    cfg.device = get_device()
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: ' f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]
    model = build_train_model(cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=distributed, validate=(not cfg.only_train), timestamp=timestamp,  meta=meta,)


if __name__ == '__main__':
    main(sys.argv[1:])
