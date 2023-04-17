# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Modification of config and checkpoint to support legacy models

import argparse
import os
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import sys
import time
import os.path as osp
from mmdet.utils import get_root_logger
from mmdet.utils import collect_env
from mmdet.apis import set_random_seed

def update_configs_with_eval_paths(cfg):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # create log dir
    log_file = osp.join(cfg.work_dir, f'evaluation_logs_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    logger.info(f'Config:\n{cfg.pretty_text}')
    deterministic = False
    logger.info(f'Set random seed to {cfg.seed}, deterministic: 'f'{deterministic}')
    set_random_seed(cfg.seed, deterministic=deterministic)
    #
    panop_eval_temp_folder =  cfg.evaluation.panop_eval_temp_folder
    logger.info(f'Evaluation results will be saved at: {panop_eval_temp_folder}')
    # generating checkpoint file path
    checkpoint_file_path = os.path.join(cfg.checkpoint_path, 'latest.pth')
    logger.info(f'Evaluation will be done for the model {checkpoint_file_path}')
    json_file = osp.join(cfg.work_dir, f'evaluation_results_{timestamp}.json')
    logger.info(f'Final evaluation results JSON file will be saved at: {json_file}')
    pathDict = {}
    pathDict['checkpoint_file_path'] = checkpoint_file_path
    return pathDict, logger, json_file

def update_legacy_cfg(cfg):
    cfg.data.test.pipeline[1]['img_scale'] = tuple(cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg

formatOnlyStr = 'Format the output results without perform evaluation. It is useful when you want to format the result to a specific format and submit it to the test server'
evalStr = 'evaluation metrics, which depends on the dataset, e.g., "mIoU"' ' for generic datasets, and "cityscapes" for Cityscapes'
def parse_args(args):
    parser = argparse.ArgumentParser( description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument( '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--format-only', action='store_true',  help=formatOnlyStr )
    parser.add_argument('--eval', type=str, nargs='+', help=evalStr)
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument( '--show-dir', help='directory where painted images will be saved')
    parser.add_argument( '--gpu-collect', action='store_true',  help='whether to use gpu to collect results.')
    parser.add_argument( '--tmpdir', help='tmp directory used for collecting results from multiple '  'workers, available when gpu_collect is not specified')
    parser.add_argument(  '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument( '--eval-options',  nargs='+', action=DictAction, help='custom options for evaluation')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument( '--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

assertStr = ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
def main(args):
    args = parse_args(args)
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [ 0.5, 0.75, 1.0, 1.25, 1.5, 1.75 ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader( dataset, samples_per_gpu=1,  workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False)
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    pathDict, logger, json_file = update_configs_with_eval_paths(cfg)
    checkpoint = load_checkpoint( model, pathDict['checkpoint_file_path'],  map_location='cpu', revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        eval_kwargs = {}
        eval_kwargs['eval_kwargs'] = cfg['evaluation']
        outputs = single_gpu_test(model, data_loader, show=False,out_dir=None, efficient_test=False, opacity=0.5, logger=logger, **eval_kwargs)
    else:
        model = MMDistributedDataParallel(  model.cuda(), device_ids=[torch.cuda.current_device()],  broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, efficient_test)
    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {}
        kwargs['logger'] = logger
        ignore_key_list = ['interval']
        for key in cfg['evaluation'].keys():
            if key not in ignore_key_list:
                kwargs[key] = cfg['evaluation'][key]
        metric = dataset.evaluate(outputs, **kwargs)
        logger.info(metric)
        metric_dict = dict(config=args.config, metric=metric)
        if json_file is not None and rank == 0:
            mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main(sys.argv[1:])
