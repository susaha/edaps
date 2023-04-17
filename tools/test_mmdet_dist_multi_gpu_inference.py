# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test_uda, multi_gpu_inference
from mmseg.datasets import (build_dataloader, build_dataset)
from mmdet.datasets import (replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root)
from tools.parser_argument_help_str import str1, str2, str3, str4, str5, str6
from tools.panoptic_deeplab.utils import create_panop_eval_folders
import sys
from mmdet.utils import get_root_logger


'''
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
python tools/test.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc.py checkpoints/SOME_CHECKPOINT.pth 
'''

def parse_args(args):
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--fuse-conv-bn', action='store_true',  help='Whether to fuse conv and bn, this will slightly increase' 'the inference speed')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='(Deprecated, please use --gpu-id) ids of gpus to use ''(only applicable to non-distributed training)')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use ' '(only applicable to non-distributed testing)')
    parser.add_argument('--format-only', action='store_true', help='Format the output results without perform evaluation. It is' 'useful when you want to format the result to a specific format and ''submit it to the test server')
    parser.add_argument('--eval', type=str,  nargs='+', help='evaluation metrics, which depends on the dataset, e.g., "bbox",' ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr',  type=float, default=0.3, help='score threshold (default: 0.3)')
    parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results.')
    parser.add_argument('--tmpdir', help='tmp directory used for collecting results from multiple ' 'workers, available when gpu-collect is not specified')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help=str1)
    parser.add_argument('--options', nargs='+', action=DictAction, help=str2)
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help=str3)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.eval_options:
        raise ValueError(str4)
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main(args):
    args = parse_args(args)

    # assert args.out or args.eval or args.format_only or args.show or args.show_dir, str5 # original i commented

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(str6)
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()

    # init distributed env first, since logger depends on the dist info.
    # if args.launcher == 'none':
    if cfg.launcher == None:
        distributed = False
    else:
        distributed = True
        init_dist(cfg.launcher, **cfg.dist_params)
        # init_dist(args.launcher, **cfg.dist_params)

    assert distributed, 'inference has to be multi gpu distr '

    test_dataloader_default_args = dict( samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.val, dict):
        cfg.data.val.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
    elif isinstance(cfg.data.val, list):
        for ds_cfg in cfg.data.val:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.val:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()

    exp_sub = 'experiments/daformer_panoptic_experiments'
    exp_root = '/media/suman/CVLHDD/apps'
    # generating the evaluation logger
    str_paths = cfg.work_dir.split('/')
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    from mmdet.models.builder import build_train_model
    model = build_train_model(cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # create log dir
    log_dir = osp.join(exp_root, exp_sub, str_paths[1], str_paths[2])
    mmcv.mkdir_or_exist(osp.abspath(log_dir))
    log_file = osp.join(log_dir, f'evaluation_logs_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    checkpoint_path = cfg.checkpoint_path
    logger.info('The following checkpoints will be evaluated ...')

    logger.info(checkpoint_path)
    eval_out_path = os.path.join(exp_root, exp_sub, checkpoint_path)
    # generating the panoptic evaluation folders
    panop_eval_folder = os.path.join(eval_out_path, 'panoptic_eval')
    panop_eval_temp_folder = create_panop_eval_folders(panop_eval_folder)
    cfg['evaluation']['panop_eval_folder'] = panop_eval_folder
    cfg['evaluation']['panop_eval_temp_folder'] = panop_eval_temp_folder
    # cfg['evaluation']['gt_dir_panop'] = eval_cfg_gt_dir_panop
    cfg['evaluation']['debug'] = cfg['debug']
    cfg['evaluation']['out_dir'] = os.path.join(panop_eval_temp_folder, 'visuals')
    logger.info(f'Evaluation results will be saved at: {panop_eval_temp_folder}')
    # generating checkpoint file path
    checkpoint_file_path = os.path.join(eval_out_path, 'latest.pth')
    logger.info(f'Evaluation will be done for the model {checkpoint_file_path}')
    # generating the json file path where the final evaluation results will be saved  # TODO
    json_file = osp.join(panop_eval_temp_folder, f'evaluation_results_{timestamp}.json')
    logger.info(f'Final evaluation results JSON file will be saved at: {json_file}')
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    checkpoint = load_checkpoint(model, checkpoint_file_path, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model = build_ddp(model, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])], broadcast_buffers=False)
    multi_gpu_inference( model, data_loader, args.tmpdir, args.gpu_collect or cfg.evaluation.get('gpu_collect', False))


if __name__ == '__main__':
    main(sys.argv[1:])
