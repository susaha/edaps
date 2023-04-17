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
from mmdet.apis import multi_gpu_test, single_gpu_test_uda, single_gpu_test_uda_for_visual_debug, single_gpu_test_uda_dump_results_to_disk
from mmseg.datasets import (build_dataloader, build_dataset)
from mmdet.datasets import (replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root)
from tools.parser_argument_help_str import str1, str2, str3, str4, str5, str6
import sys
from mmdet.utils import get_root_logger
from mmdet.models.builder import build_train_model
from mmdet.utils import collect_env
from mmdet.apis import set_random_seed

def parse_args(args):
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
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
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
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
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_train_model(cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
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
    checkpoint = None
    outputs = None
    if cfg.checkpoint_path:
        checkpoint_path = cfg.checkpoint_path
        logger.info('The following checkpoints will be evaluated ...')
        logger.info(checkpoint_path)
        # generating checkpoint file path
        checkpoint_file_path = os.path.join(checkpoint_path, 'latest.pth')
        logger.info(f'Evaluation will be done for the model {checkpoint_file_path}')
        checkpoint = load_checkpoint(model, checkpoint_file_path, map_location='cpu')

    panop_eval_temp_folder = cfg['evaluation']['panop_eval_temp_folder']
    logger.info(f'Evaluation results will be saved at: {panop_eval_temp_folder}')

    # generating the json file path where the final evaluation results will be saved
    json_file = osp.join(cfg.work_dir, f'evaluation_results_{timestamp}.json')
    logger.info(f'Final evaluation results JSON file will be saved at: {json_file}')
    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if cfg.checkpoint_path and checkpoint:
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        if cfg.generate_only_visuals_without_eval:
            outputs = single_gpu_test_uda_for_visual_debug(
                model,
                data_loader,
                show=args.show,
                out_dir=args.show_dir,
                debug=cfg.debug,
                show_score_thr=args.show_score_thr,
                dataset_name=cfg.evaluation.dataset_name,
                panop_eval_temp_folder=panop_eval_temp_folder,
                )
        elif cfg.dump_predictions_to_disk:
            dump_path = os.path.join(cfg.work_dir, 'results_numpys')
            os.makedirs(dump_path,exist_ok=True)
            outputs = single_gpu_test_uda_dump_results_to_disk(
                model,
                data_loader,
                out_dir=dump_path,
                debug=cfg.debug,
                dataset_name=cfg.evaluation.dataset_name,
                logger=logger,
            )
        elif cfg.evaluate_from_saved_png_predictions:
            # In this case we assume that all the prediction pngs for semantic, instance and panoptics have been already saved to disk
            # by the panoptic deeplab evaluation scripts under panop_eval_temp_folder/. So, we call the evalute function below.
            pass
        else:
            outputs = single_gpu_test_uda(
                model,
                data_loader,
                show=args.show,
                out_dir=args.show_dir,
                debug=cfg.debug,
                show_score_thr=args.show_score_thr,
                dataset_name=cfg.evaluation.dataset_name
            )

    else:
        model = build_ddp(model, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])], broadcast_buffers=False)
        outputs = multi_gpu_test( model, data_loader, args.tmpdir, args.gpu_collect or cfg.evaluation.get('gpu_collect', False))

    if cfg.generate_only_visuals_without_eval:
        pass

    elif cfg.evaluate_from_saved_png_predictions and not outputs:
        cfg['evaluation']['panop_eval_temp_folder'] = cfg['panop_eval_temp_folder_previous']
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'dynamic_intervals']:
            eval_kwargs.pop(key, None)
        metric = dataset.evaluate(outputs, logger=logger, **eval_kwargs)
        logger.info(metric)
        metric_dict = dict(config=args.config, metric=metric)
        if json_file is not None and rank == 0:
            mmcv.dump(metric_dict, json_file)

    elif cfg.dump_predictions_to_disk and outputs:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'dynamic_intervals']:
            eval_kwargs.pop(key, None)
        metric = dataset.evaluate(outputs, logger=logger, **eval_kwargs)
        logger.info(metric)
        metric_dict = dict(config=args.config, metric=metric)
        if json_file is not None and rank == 0:
            mmcv.dump(metric_dict, json_file)
    else:
        if outputs:
            rank, _ = get_dist_info()
            if rank == 0:
                if args.out:
                    print(f'\nwriting results to {args.out}')
                    mmcv.dump(outputs, args.out)
                kwargs = {} if args.eval_options is None else args.eval_options
                if args.format_only:
                    dataset.format_results(outputs, **kwargs)
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best','rule', 'dynamic_intervals' ]:
                    eval_kwargs.pop(key, None)
                # main evaluation
                metric = dataset.evaluate(outputs, logger=logger, **eval_kwargs)
                logger.info(metric)
                metric_dict = dict(config=args.config, metric=metric)
                if json_file  is not None and rank == 0:
                    mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main(sys.argv[1:])
