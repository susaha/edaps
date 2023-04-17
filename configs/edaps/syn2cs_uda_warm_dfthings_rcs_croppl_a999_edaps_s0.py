# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

debug = False
_base_ = [
		"../_base_/default_runtime_mmdet_mr.py",
		"../_base_/models/maskrcnn_sepaspp_mitb5.py",
		"../_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py",
		"../_base_/uda/dacs_a999_fdthings.py",
		"../_base_/schedules/adamw.py",
		"../_base_/schedules/poly10warm.py"
]

n_gpus = 1
gpu_mtotal = 23000
total_train_time =  "21:00:00"
n_cpus = 16
mem_per_cpu = 16000
machine = "local"
euler_template_fname = "euler_template_slurm_syn2city.sh"
resume_from = None
load_from = None
only_eval = False
only_train = False
activate_auto_scale_lr = False
auto_scale_lr = dict(enable=False, base_batch_size=16)
print_layer_wise_lr = False
file_sys = "Slurm"
launcher = None
generate_only_visuals_without_eval = False
seed = 0
uda = dict(
        pseudo_weight_ignore_top=15,
        pseudo_weight_ignore_bottom=120,
        share_src_backward=True,
        debug_img_interval=1000,
        imnet_feature_dist_lambda=0.005,
        alpha=0.999,
        pseudo_threshold=0.968,
        disable_mix_masks=False
)

data = dict(
	    samples_per_gpu=2, # batchsize (2 source  + 2 target images)
        workers_per_gpu=4,
        train=dict(
                rare_class_sampling=dict(
                                        min_pixels=3000,
                                        class_temp=0.01,
                                        min_crop_ratio=0.5
                                        )
                ),
        val=dict(
                ann_dir="gtFine_panoptic/cityscapes_panoptic_val",
                data_root="data/cityscapes"
        )
)
optimizer_config =  None

optimizer = dict(
        lr=6e-05,
        paramwise_cfg=dict(
            custom_keys=dict(
                neck=dict(lr_mult=10.0), head=dict(lr_mult=10.0), pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0)
            )
        )
)

evaluation = dict(
    interval=40000,
    metric=["mIoU", "mPQ", "mAP"],
    eval_type="maskrcnn_panoptic",
    dataset_name="cityscapes",
    gt_dir="data/cityscapes/gtFine/val",
    gt_dir_panop="data/cityscapes/gtFine_panoptic",
    num_samples_debug=12,
    post_proccess_params=dict(
        num_classes=19,
        ignore_label=255,
        mapillary_dataloading_style="OURS",
        label_divisor=1000,
        train_id_to_eval_id=[7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0],
        thing_list=[11, 12, 13, 14, 15, 16, 17, 18],
        mask_score_threshold=0.95,
        debug=False,
        dump_semantic_pred_as_numpy_array=False,
        load_semantic_pred_as_numpy_array=False,
        semantic_pred_numpy_array_location=None,
        use_semantic_decoder_for_instance_labeling=False,
        use_semantic_decoder_for_panoptic_labeling=False,
        nms_th=None,
        intersec_th=None,
        upsnet_mask_pruning=False,
        generate_thing_cls_panoptic_from_instance_pred=False
    ),
    visuals_pan_eval=False,
    evalScale="2048x1024",
    panop_eval_folder="work_dirs/local-exp1/230401_1734_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ffbee/panoptic_eval",
    panop_eval_temp_folder="work_dirs/local-exp1/230401_1734_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ffbee/panoptic_eval/panop_eval_01-04-2023_17-34-49-423139",
    debug=False,
    out_dir="work_dirs/local-exp1/230401_1734_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ffbee/panoptic_eval/panop_eval_01-04-2023_17-34-49-423139/visuals"
)

runner = dict(
    type="IterBasedRunner",
    max_iters= 40000
)

checkpoint_config = dict(
    by_epoch=False,
    interval=40000,
    max_keep_ckpts=1
)

log_config = dict(
    interval=20,
)
name =  "syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0"
exp = 1
exp_root = "edaps_experiments"
exp_sub = "exp-00001"
name_dataset =  "synthia2cityscapes",
name_architecture =  "maskrcnn_mitb5",
name_encoder =  "mitb5",
name_decoder =  "maskrcnn",
name_uda =  "dacs_rcs0.01_cpl",
name_opt =  "adamw_6e-05_pmTrue_poly10warm_1x1_40k",
work_dir =  "work_dirs/local-exp1/syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0",
git_rev =  "41390479d2943fdece2e7362b66f9a773d3f42c3"

