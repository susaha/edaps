# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Modification of config and checkpoint to support legacy models

'''
$ cddaformer (cd to daformer_panoptic folder)
$ srcdaformer (activate the daformer python environment)
$ PYTHONPATH="/home/suman/apps/code/CVPR2022/daformer_panoptic:$PYTHONPATH" && export PYTHONPATH
'''

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
from argparse import Namespace
import json
from types import SimpleNamespace
import time
import os.path as osp
from mmseg.utils import collect_env, get_root_logger
from tools.panoptic_deeplab.utils import create_panop_eval_folders


# user inputs
# exp_name = '220425_1752_syn2cs_dacs_panoptic_a999_fdthings_rcs001_cpl_daformer_panoptic_sepaspp_mitb5_poly10warm_s0_8e844'
# base_path = f'/media/suman/CVLHDD/apps/experiments/daformer_panoptic_experiments/euler-exp1004/{exp_name}'
# CONFIG_FILE = f'{base_path}/{exp_name}.json'
# CHECKPOINT_FILE = f'{base_path}/latest.pth'
# SHOW_DIR = "/media/suman/CVLHDD/apps/experiments/daformer_panoptic_experiments/euler-exp1004/" \
#            "220425_1752_syn2cs_dacs_panoptic_a999_fdthings_rcs001_cpl_daformer_panoptic_sepaspp_mitb5_poly10warm_s0_8e844/preds/"
# SHOW_DIR = None
# EVAL='mIoU'
# OPACITY=1.0

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.val.pipeline[1]['img_scale'] = tuple(cfg.data.val.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


# def parse_args(args):
    # parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('--train_json', help='test config file path')
    # parser.add_argument('--checkpoint', help='checkpoint file')
    # parser.add_argument('--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    # parser.add_argument('--out', help='output result file in pickle format')
    # parser.add_argument('--format-only', action='store_true', help='Format the output results without perform evaluation. It is useful when you want to format the result to a specific format and submit it to the test server')
    # parser.add_argument('--eval',type=str, nargs='+', help='evaluation metrics, which depends on the dataset, e.g., "mIoU" for generic datasets, and "cityscapes" for Cityscapes')
    # parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument('--show-dir', help='directory where painted images will be saved')
    # parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results.')
    # parser.add_argument('--tmpdir', help='tmp directory used for collecting results from multiple workers, available when gpu_collect is not specified')
    # parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    # parser.add_argument('--eval-options', nargs='+', action=DictAction, help='custom options for evaluation')
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    # parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
    # parser.add_argument('--local_rank', type=int, default=0)
    # args = parser.parse_args(args)
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    # return args



def main(args):

    print('************')
    print(args)
    with open(args[0], 'r') as f:
        data = json.load(f)
    args = Namespace(**data)

    # assert args.out or args.eval or args.format_only or args.show \
    #     or args.show_dir, \
    #     ('Please specify at least one operation (save/eval/format/show the '
    #      'results / save the results) with the argument "--out", "--eval"'
    #      ', "--format-only", "--show" or "--show-dir"')

    # # remove here the eval folder
    panop_eval_temp_folder = args.panop_eval_temp_folder
    panop_eval_folder = args.panop_eval_folder
    panop_eval_temp_folder_name = args.panop_eval_temp_folder_name

    print('removing panop_eval_temp_folder: {}'.format(panop_eval_temp_folder))
    strCmd2 = 'rm -r ' + ' ' + panop_eval_temp_folder
    os.system(strCmd2)
    print('creating panop_eval_temp_folder: {}'.format(panop_eval_temp_folder))
    panop_eval_temp_folder = create_panop_eval_folders(panop_eval_folder, panop_eval_temp_folder=panop_eval_temp_folder_name)

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)

    # cfg['data']['val']['data_root'] = '../data/cityscapes/'

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.val.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.val.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.val.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0, # cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
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

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.exp_path, f'{timestamp}_panoptic_eval.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'log file: {log_file}')


    eval_kwargs = args.eval_kwargs
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, efficient_test, args.opacity, logger=logger, **eval_kwargs)
    else:
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        eval_kwargs['eval_kwargs'].pop('interval')
        kwargs = eval_kwargs['eval_kwargs']
        kwargs['logger'] = logger
        # kwargs = {} if args.eval_options is None else args.eval_options # original


        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, **kwargs)
            # dataset.evaluate(outputs, args.eval, **kwargs) # original


if __name__ == '__main__':
    main(sys.argv[1:])
