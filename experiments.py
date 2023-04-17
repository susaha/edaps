# ---------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------------------------------

import itertools


def set_semantic_and_instance_loss_weights(cfg, loss_weight_semanitc, loss_weight_instance):
    cfg.setdefault('model', {})
    # daformer semantic head
    cfg['model'].setdefault('decode_head', {})
    cfg['model']['decode_head'].setdefault('loss_decode', {})
    cfg['model']['decode_head']['loss_decode']['loss_weight'] = loss_weight_semanitc
    # mask-rcnn instance head
    cfg['model'].setdefault('rpn_head', {})
    cfg['model']['rpn_head'].setdefault('loss_cls', {})
    cfg['model']['rpn_head'].setdefault('loss_bbox', {})
    cfg['model']['rpn_head']['loss_cls']['loss_weight'] = loss_weight_instance
    cfg['model']['rpn_head']['loss_bbox']['loss_weight'] = loss_weight_instance
    cfg['model'].setdefault('roi_head', {})
    cfg['model']['roi_head'].setdefault('bbox_head', {})
    cfg['model']['roi_head']['bbox_head'].setdefault('loss_cls', {})
    cfg['model']['roi_head']['bbox_head'].setdefault('loss_bbox', {})
    cfg['model']['roi_head']['bbox_head']['loss_cls']['loss_weight'] = loss_weight_instance
    cfg['model']['roi_head']['bbox_head']['loss_bbox']['loss_weight'] = loss_weight_instance
    cfg['model']['roi_head'].setdefault('mask_head', {})
    cfg['model']['roi_head']['mask_head'].setdefault('loss_mask', {})
    cfg['model']['roi_head']['mask_head']['loss_mask']['loss_weight'] = loss_weight_instance
    if loss_weight_semanitc == 0:
        cfg['evaluation']['metric'] = ['mAP']
    else:
        cfg['evaluation']['metric'] = ['mIoU']
    return cfg


def get_default_runtime_base():
    return '_base_/default_runtime_mmdet_mr.py'


def get_model_base_dacs(architecture, semantic_decoder, backbone, uda_model_type, ):
    if uda_model_type == 'dacs':
        dacs_model_base = f'_base_/models/{architecture}_{semantic_decoder}_{backbone}.py'
    elif 'dacs_inst' in uda_model_type:
        dacs_model_base = f'_base_/models/{architecture}_{semantic_decoder}_{backbone}_dacsInst.py'
    else:
        raise NotImplementedError(f'No impl found for uda_model_type: {uda_model_type}')
    return dacs_model_base


def get_model_base(architecture, backbone, uda, semantic_decoder='sepaspp', uda_model_type='dacs', ):
    dacs_model_base = None
    if uda == 'dacs':
        dacs_model_base = get_model_base_dacs(architecture, semantic_decoder, backbone, uda_model_type, )
    return {
            'target-only': f'_base_/models/{architecture}_{semantic_decoder}_{backbone}.py',
            'source-only': f'_base_/models/{architecture}_{semantic_decoder}_{backbone}.py',
            'dacs':        dacs_model_base
    }[uda]


def get_dataset_base_dacs(include_diffusion_data, source, target, evalScale):
    if not include_diffusion_data:
        if evalScale:
            dacs_dataset_base = f'_base_/datasets/uda_{source}_to_{target}_maskrcnn_panoptic_evalScale_{evalScale}.py'
        else:
            dacs_dataset_base = f'_base_/datasets/uda_{source}_to_{target}_maskrcnn_panoptic.py'
    else:
        dacs_dataset_base = f'_base_/datasets/uda_{source}_to_{target}_maskrcnn_panoptic_diffusion.py'
    return dacs_dataset_base


def get_dataset_base(uda, source, target, include_diffusion_data=False, evalScale=None):
    if uda == 'dacs':
        dacs_dataset_base = get_dataset_base_dacs(include_diffusion_data, source, target, evalScale)
    return {
            'target-only': f'_base_/datasets/{uda}_{target}_maskrcnn_panoptic.py',
            'source-only': f'_base_/datasets/{uda}_{source}_to_{target}_maskrcnn_panoptic.py',
            'dacs':        dacs_dataset_base
    }[uda]


def get_uda_base(uda_sub_type, uda_model_type='dacs'):
    if uda_model_type == 'dacs':
        uda_model = 'dacs'
    elif uda_model_type == 'dacs_inst':
        uda_model = 'dacs_inst'
    elif uda_model_type == 'dacs_inst_v2':
        uda_model = 'dacs_inst_v2'
    return f'_base_/uda/{uda_model}_{uda_sub_type}.py'


def get_optimizer_base(opt):
    return f'_base_/schedules/{opt}.py'


def get_schedule_base(schedule):
    return f'_base_/schedules/{schedule}.py'


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def get_eval_params(mask_score_threshold, debug, mapillary_dataloading_style,
                    semantic_pred_numpy_array_location=None,
                    dump_semantic_pred_as_numpy_array=False,
                    load_semantic_pred_as_numpy_array=False,
                    use_semantic_decoder_for_instance_labeling=False,
                    use_semantic_decoder_for_panoptic_labeling=False,
                    nms_th=None,
                    intersec_th=None,
                    upsnet_mask_pruning=False,
                    generate_thing_cls_panoptic_from_instance_pred=False,
                    ):

    train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0]
    thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
    panop_deeplab_eval_post_process_params = dict(num_classes=19,
                                                  ignore_label=255,
                                                  mapillary_dataloading_style=mapillary_dataloading_style,
                                                  label_divisor=1000,
                                                  train_id_to_eval_id=train_id_to_eval_id,
                                                  thing_list=thing_list,
                                                  mask_score_threshold=mask_score_threshold,
                                                  debug=debug,
                                                  dump_semantic_pred_as_numpy_array=dump_semantic_pred_as_numpy_array,
                                                  load_semantic_pred_as_numpy_array=load_semantic_pred_as_numpy_array,
                                                  semantic_pred_numpy_array_location=semantic_pred_numpy_array_location,
                                                  use_semantic_decoder_for_instance_labeling=use_semantic_decoder_for_instance_labeling,
                                                  use_semantic_decoder_for_panoptic_labeling=use_semantic_decoder_for_panoptic_labeling,
                                                  nms_th=nms_th,
                                                  intersec_th=intersec_th,
                                                  upsnet_mask_pruning=upsnet_mask_pruning,
                                                  generate_thing_cls_panoptic_from_instance_pred=generate_thing_cls_panoptic_from_instance_pred,
                                                  )
    return panop_deeplab_eval_post_process_params


def generate_experiment_cfgs(id, machine_name):
    def get_initial_cfg():
        return {
            'debug': debug,
            '_base_': [],
            'n_gpus': n_gpus,
            'gpu_mtotal': gpu_mtotal,
            'total_train_time': total_train_time,
            'n_cpus': n_cpus,
            'mem_per_cpu': mem_per_cpu,
            'machine': machine,
            'resume_from': resume_from,
            'load_from': load_from,
            'only_eval': only_eval,
            'only_train': only_train,
            'activate_auto_scale_lr': activate_auto_scale_lr,
            'auto_scale_lr': dict(enable=activate_auto_scale_lr, base_batch_size=16),
            'print_layer_wise_lr': print_layer_wise_lr,
            'file_sys': file_sys,
            'launcher': launcher,
            'generate_only_visuals_without_eval': generate_only_visuals_without_eval,
            'dump_predictions_to_disk': dump_predictions_to_disk,
            'evaluate_from_saved_png_predictions': evaluate_from_saved_png_predictions,
            'panop_eval_temp_folder_previous': panop_eval_temp_folder_previous,
            'exp_sub': exp_sub,
            'exp_root': exp_root,
        }

    def config_from_vars():
        cfg = get_initial_cfg()
        # get default runtime base config
        cfg['_base_'].append(get_default_runtime_base())
        # set seed
        if seed is not None:
            cfg['seed'] = seed
        # get model base config
        cfg['_base_'].append(get_model_base(architecture, backbone, uda,
                                            semantic_decoder=semantic_decoder,
                                            uda_model_type=uda_model_type,
                                            )
                             )
        # get dataset base config
        cfg['_base_'].append(get_dataset_base(uda, source, target,
                                              include_diffusion_data=include_diffusion_data,
                                              evalScale=evalScale,
                                               )
                             )
        # get uda base config
        if 'dacs' in uda:
            cfg['_base_'].append(get_uda_base(uda_sub_type, uda_model_type=uda_model_type))
        #
        if 'dacs' in uda and plcrop:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        if 'dacs' in uda and not plcrop:
            cfg.setdefault('uda', {})
        if 'dacs' in uda:
            cfg['uda']['share_src_backward'] = share_src_backward
            cfg['uda']['debug_img_interval'] = debug_img_interval
            cfg['uda']['imnet_feature_dist_lambda'] = imnet_feature_dist_lambda
            cfg['uda']['alpha'] = mean_teacher_alpha
            cfg['uda']['pseudo_threshold'] = pseudo_threshold
            cfg['uda']['disable_mix_masks'] = disable_mix_masks
            if 'dacs_inst' in uda_model_type:
                cfg['uda']['activate_uda_inst_losses'] = activate_uda_inst_losses
                cfg['uda']['mix_masks_only_thing'] = mix_masks_only_thing
                cfg['uda']['inst_pseduo_weight'] = inst_pseduo_weight
                cfg['uda']['swtich_off_mix_sampling'] = swtich_off_mix_sampling  # NOT IN USE
                cfg['uda']['switch_off_self_training'] = switch_off_self_training # NOT IN USE
        cfg['data'] = dict( samples_per_gpu=batch_size, workers_per_gpu=workers_per_gpu, train={})
        # setup config for RCS
        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)
        # Setup the ann_dir for validation
        cfg['data'].setdefault('val', {})
        cfg['data']['val']['ann_dir'] = ann_dir
        cfg['data']['val']['data_root'] = data_root
        if include_diffusion_data:
            # cfg.setdefault('data', {}).setdefault('train', {})
            cfg['data'].setdefault('train', {}).setdefault('target', {})
            cfg['data']['train']['target']['include_diffusion_data'] = include_diffusion_data
            cfg['data']['train']['target']['diffusion_set'] = diffusion_set

        # Setup optimizer
        # if 'dacs' in uda:
        cfg['optimizer_config'] = None  # Don't use outer optimizer
        # get optimizer base config
        cfg['_base_'].append(get_optimizer_base(opt))
        # get schedule base config
        cfg['_base_'].append(get_schedule_base(schedule))
        # set the learing rate of the backbone to lr
        # if pmult is True then set the learing rate of the neck and heads to lr*10.0
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            if set_diff_pmult_for_sem_and_inst_heads:
                assert pmult_inst_head, 'pmult_inst_head can not be None!'
                opt_param_cfg['decode_head'] = dict(lr_mult=10.) # semantic head
                opt_param_cfg['neck'] = dict(lr_mult=pmult_inst_head)  # this for the FPN
                opt_param_cfg['rpn_head'] = dict(lr_mult=pmult_inst_head)
                opt_param_cfg['roi_head'] = dict(lr_mult=pmult_inst_head)
            else:
                opt_param_cfg['neck'] = dict(lr_mult=10.)  # this for the FPN
                opt_param_cfg['head'] = dict(lr_mult=10.) # all heads: decode-head, fpn-head, roi-head

        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)
        # Set evaluation configs

        if use_class_specific_mask_score_th:
            mask_score_threshold_dynamic = mask_score_threshold_class_specific
        else:
            mask_score_threshold_dynamic = mask_score_threshold

        cfg['evaluation'] = dict(interval=eval_interval, metric=eval_metric_list,
                                 eval_type=eval_type, dataset_name=target,
                                 gt_dir=gt_dir_instance, gt_dir_panop=gt_dir_panoptic, num_samples_debug=num_samples_debug,
                                 post_proccess_params=get_eval_params(mask_score_threshold_dynamic, debug,
                                                                      mapillary_dataloading_style=mapillary_dataloading_style,
                                                                      semantic_pred_numpy_array_location=semantic_pred_numpy_array_location,
                                                                      dump_semantic_pred_as_numpy_array=dump_semantic_pred_as_numpy_array,
                                                                      load_semantic_pred_as_numpy_array=load_semantic_pred_as_numpy_array,
                                                                      use_semantic_decoder_for_instance_labeling=use_semantic_decoder_for_instance_labeling,
                                                                      use_semantic_decoder_for_panoptic_labeling=use_semantic_decoder_for_panoptic_labeling,
                                                                      nms_th=nms_th,
                                                                      intersec_th=intersec_th,
                                                                      upsnet_mask_pruning=upsnet_mask_pruning,
                                                                      generate_thing_cls_panoptic_from_instance_pred=generate_thing_cls_panoptic_from_instance_pred,
                                                                      ),
                                 visuals_pan_eval=dump_visuals_during_eval,
                                 evalScale=evalScale,
                                 evaluate_from_saved_numpy_predictions=evaluate_from_saved_numpy_predictions,
                                 evaluate_from_saved_png_predictions=evaluate_from_saved_png_predictions,

                                 )
        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(by_epoch=False, interval=checkpoint_interval, max_keep_ckpts=1)
        # Set the log_interval
        cfg['log_config'] = dict(interval=log_interval)
        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
        if 'dacs' in uda and plcrop:
            uda_mod += '_cpl'
        cfg['name'] = f'{source}2{target}_{uda_mod}_{architecture}_' + f'{backbone}_{schedule}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' + f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') .replace('False', 'F').replace('cityscapes', 'cs').replace('synthia', 'syn')
        # returning the config for a single experiment
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    debug = False
    machine = machine_name
    iters = 40000
    interval = iters
    interval_debug = 3
    uda = 'dacs'
    data_root = 'data/cityscapes'
    # ----------------------------------------
    # --- Set the debug time configs ---
    # ----------------------------------------
    n_gpus = 1 if debug else 1
    batch_size = 1 if debug else 2  # samples_per_gpu
    workers_per_gpu = 0 if debug else 4 # if 'dacs' in uda else 2
    eval_interval = interval_debug if debug else interval
    checkpoint_interval = interval_debug if debug else interval
    ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
    log_interval = 1 if debug else 50
    debug_img_interval = 1 if debug else 5000
    # ----------------------------------------
    architecture = 'maskrcnn'
    backbone = 'mitb5'
    models = [(architecture, backbone)]
    udas = [uda]
    uda_sub_type = 'a999_fdthings'
    source, target = 'synthia', 'cityscapes'
    datasets = [(source, target)]
    seed = 0
    plcrop = True
    rcs_T = 0.01
    imnet_feature_dist_lambda = 0.005
    opt = 'adamw'
    schedule = 'poly10warm'
    lr = 0.00006
    pmult = True
    only_train = False
    only_eval = False
    eval_type = 'maskrcnn_panoptic'
    resume_from = None
    load_from = None
    activate_auto_scale_lr = False
    print_layer_wise_lr = False
    share_src_backward = True
    uda_model_type = 'dacs'
    activate_uda_inst_losses = False
    mix_masks_only_thing = False
    inst_pseduo_weight = None
    num_samples_debug = 12
    gt_dir_instance = 'data/cityscapes/gtFine/val'
    gt_dir_panoptic = 'data/cityscapes/gtFine_panoptic'
    eval_metric_list = ['mIoU', 'mPQ', 'mAP']
    mapillary_dataloading_style = 'OURS'
    set_diff_pmult_for_sem_and_inst_heads = False
    semantic_decoder = 'sepaspp'
    dump_semantic_pred_as_numpy_array = False
    load_semantic_pred_as_numpy_array = False
    semantic_pred_numpy_array_location = None
    mask_score_threshold = 0.95
    mask_score_threshold_class_specific = None
    use_class_specific_mask_score_th = False
    use_semantic_decoder_for_instance_labeling = False  # Not in use
    use_semantic_decoder_for_panoptic_labeling = False  # Not in use
    launcher = None
    upsnet_mask_pruning = False
    generate_thing_cls_panoptic_from_instance_pred = False
    nms_th = None
    intersec_th = None
    generate_only_visuals_without_eval = False
    dump_predictions_to_disk = False
    # diffusion data
    include_diffusion_data = False
    diffusion_set = None
    pmult_inst_head = None
    evalScale = None
    evaluate_from_saved_numpy_predictions = False
    evaluate_from_saved_png_predictions = False
    panop_eval_temp_folder_previous = None
    mean_teacher_alpha = 0.999
    pseudo_threshold = 0.968
    disable_mix_masks = False
    # The below params are not in use
    swtich_off_mix_sampling = False
    switch_off_self_training = False
    dump_visuals_during_eval = False  # if True, save the predictions to disk at evaluation
    exp_root = "edaps_experiments"
    exp_sub = f'exp-{id:05d}'
    # override experiment folders, if they are not none, these values will be used
    # override_exp_folders = False
    # str_unique_name = None
    str_panop_eval_temp_folder = None

    # The below params are not in use
    n_cpus = 16
    mem_per_cpu = 16000
    gpu_mtotal = 23000
    total_train_time = '24:00:00'
    file_sys = 'Slurm'

    cfgs = []

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes (Table 1)
    # -------------------------------------------------------------------------
    if id == 1:
        seeds = [0,1,2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Mapillary (Table 2)
    # -------------------------------------------------------------------------
    elif id == 2:
        data_root = 'data/mapillary'
        ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls' #
        target = 'mapillary'
        num_samples_debug = 13
        gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
        gt_dir_panoptic = 'data/mapillary'
        seeds = [1, 2] # [0, 1, 2]
        plcrop = False
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)


    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes
    # source-only and target-only (oracle or supervised) models
    # (Table 3 bottom row; Table 7 top row : Source-only model)
    # -------------------------------------------------------------------------
    elif id == 4:
        udas = [
            'source-only',
            'target-only',
        ]
        seeds = [0,1,2]
        for seed, uda in  itertools.product(seeds, udas):
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # M-Net: SYNTHIA → Cityscapes (Table 5)
    # M-Net training and evaluation are done in 4 stages:
    # Stage-1: Train the semantic segmentation network (id=50)
    # Stage-2: Train the instance segmentation network (id=51)
    # Stage-3: Extract the semantic segmentation predictions (id=52)
    # Stage-4: Extract the instance segmentation predictions and
    # evaluate the M-Net (id=53)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Stage-1: M-Net: SYNTHIA → Cityscapes (Table 5)
    # Train the semantic segmentation network
    # -------------------------------------------------------------------------
    elif id == 50:
        seeds = [0, 1, 2]
        loss_weight_semanitc, loss_weight_instance = 1.0, 0.0
        for seed in seeds:
            cfg = config_from_vars()
            cfg = set_semantic_and_instance_loss_weights(cfg, loss_weight_semanitc, loss_weight_instance)
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # Stage-2: M-Net: SYNTHIA → Cityscapes (Table 5)
    # Train the instance segmentation network
    # -------------------------------------------------------------------------
    elif id == 51:
        seeds = [0, 1, 2]
        loss_weight_semanitc, loss_weight_instance = 0.0, 1.0
        for seed in seeds:
            cfg = config_from_vars()
            cfg = set_semantic_and_instance_loss_weights(cfg, loss_weight_semanitc, loss_weight_instance)
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # Stage-3: M-Net: SYNTHIA → Cityscapes (Table 5)
    # Utilize the semantic segmentation network that has been trained in expid=50
    # to generate predictions for semantic segmentation,
    # and save the predictions as a numpy array to the disk.
    # -------------------------------------------------------------------------
    elif id == 52:
        batch_size = 1
        workers_per_gpu = 0
        dump_semantic_pred_as_numpy_array = True
        eval_metric_list = ['mIoU']
        # Put here the checkpoint locations of the semantic segmentation network that has been trained in expid=50
        # An example is given below:
        semantic_model_checkpoint_locations = [
            'path/to/the/trained/semantic/segmentation/network/model1',
            'path/to/the/trained/semantic/segmentation/network/model2',
            'path/to/the/trained/semantic/segmentation/network/model3'
        ]
        # An example:
        # semantic_model_checkpoint_locations = [
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_322b3',
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_6eb04',
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_22080',
        #     ]
        for cl in semantic_model_checkpoint_locations:
            cfg = config_from_vars()
            cfg['checkpoint_path'] = cl
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # Stage-4: M-Net: SYNTHIA → Cityscapes (Table 5)
    # Evaluate M-Net model
    # -------------------------------------------------------------------------
    elif id == 53:
        batch_size = 1
        workers_per_gpu = 0
        load_semantic_pred_as_numpy_array = True
        # Set the paths for the instance segmentation model which has been trained in expid=51
        instance_model_checkpoint_locations = [
            'path/to/the/trained/instance/segmentation/network/model1',
            'path/to/the/trained/instance/segmentation/network/model2',
            'path/to/the/trained/instance/segmentation/network/model3'
        ]
        # Set the paths for the saved smenaitc segmentation predictions
        semantic_pred_numpy_array_location_list = [
            'path/to/the/semanitc/segmentation/predictions/numpy/files/model1',
            'path/to/the/semanitc/segmentation/predictions/numpy/files/model2',
            'path/to/the/semanitc/segmentation/predictions/numpy/files/model3'
        ]
        # Examples are given below:
        # instance_model_checkpoint_locations = [
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a5398',
        # ]
        # semantic_pred_numpy_array_location_list = [
        #     '/<experiment-root-folder>/'
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_322b3/'
        #     'panoptic_eval/panop_eval_09-11-2022_15-52-53-341753/semantic'
        # ]
        for imcl, semantic_pred_numpy_array_location in zip(instance_model_checkpoint_locations, semantic_pred_numpy_array_location_list):
            cfg = config_from_vars()
            cfg['checkpoint_path'] = imcl
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes:
    # Evaluate EDAPS Model
    # -------------------------------------------------------------------------
    elif id == 6:
        seed = 0
        batch_size = 1
        workers_per_gpu = 0
        checkpoint_path = 'path/to/the/latest/checkpoint'
        cfg = config_from_vars()
        cfg['checkpoint_path'] = checkpoint_path
        cfgs.append(cfg)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes :
    # generate visualization without evaluation
    # This for the demo, just download the pretrained EDAPS model
    # save it to pretrained_edaps/
    # and run inference on the Cityscapes validation set
    # The predictions will be saved to disk
    # -------------------------------------------------------------------------
    elif id == 7:
        batch_size = 1
        workers_per_gpu = 0
        generate_only_visuals_without_eval = True
        dump_visuals_during_eval = True
        checkpoint_path = 'path/to/the/latest/checkpoint'
        cfg = config_from_vars()
        cfg['checkpoint_path'] = checkpoint_path
        cfgs.append(cfg)

    # --- RETURNING CFGS ---
    return cfgs
