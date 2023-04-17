# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Supports Mapillary Vistas dataloading for panoptic segmentation.
# ------------------------------------------------------------------------------------


from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset
import torch


@DATASETS.register_module()
class MapillaryDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train', 'val']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(MapillaryDataset, self).__init__(
                                                img_suffix='.jpg',
                                                seg_map_suffix='.png',
                                                split=None,
                                                **kwargs
                                                )


    def evaluate(self, results, metric='mIoU', logger=None, imgfile_prefix=None, efficient_test=False,
                 eval_type=None, panop_eval_folder=None, panop_eval_temp_folder=None, dataset_name=None,
                 gt_dir=None, debug=None, num_samples_debug=None, gt_dir_panop=None,
                 post_proccess_params=None, visuals_pan_eval=None, out_dir=None, evalScale=None,
                 evaluate_from_saved_numpy_predictions=None, evaluate_from_saved_png_predictions=None):

        cuda = torch.device('cuda')
        eval_results = dict()
        print(f'####### eval_type={eval_type} #######')
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            if eval_type == 'daformer':
                eval_results.update(super(MapillaryDataset,self).evaluate(results, metrics, logger, efficient_test, evalScale))
            elif eval_type == 'panop_deeplab':
                eval_results.update(
                    super(MapillaryDataset,self).evaluate_panoptic(
                    results, cuda, panop_eval_temp_folder, dataset_name, gt_dir, debug,
                        num_samples_debug, gt_dir_panop, logger, post_proccess_params,
                        visuals_pan_eval
                                                                    )
                                    )
            elif eval_type == 'maskformer': # eval mask based mIoU, mPQ, mAP
                eval_results.update(
                    super(MapillaryDataset, self).evaluate_panoptic_for_maskformer(
                        results, cuda, panop_eval_temp_folder, dataset_name, gt_dir, debug, num_samples_debug,
                        gt_dir_panop, logger, post_proccess_params, visuals_pan_eval, out_dir
                    )
                )
            elif eval_type == 'maskrcnn': # only eval inst seg. mAP
                eval_results.update(
                    super(MapillaryDataset, self).evaluate_instance_for_maskrcnn(
                        results, cuda, panop_eval_temp_folder, dataset_name, gt_dir, debug, num_samples_debug,
                        gt_dir_panop, logger, post_proccess_params, visuals_pan_eval, out_dir
                                                                                )
                )
            elif eval_type == 'maskrcnn_panoptic' and not evaluate_from_saved_png_predictions:
                eval_results.update(
                    super(MapillaryDataset, self).evaluate_panoptic_for_maskrcnn(
                        results, cuda, panop_eval_temp_folder, dataset_name, gt_dir, debug, num_samples_debug,
                        gt_dir_panop, logger, post_proccess_params, visuals_pan_eval, out_dir, metric, evalScale,
                        evaluate_from_saved_numpy_predictions
                                                                                )
                )
            elif eval_type == 'maskrcnn_panoptic' and evaluate_from_saved_png_predictions:
                eval_results.update(
                    super(MapillaryDataset, self).evaluate_panoptic_for_maskrcnn_v2(
                        results, cuda, panop_eval_temp_folder, dataset_name, gt_dir, debug, num_samples_debug,
                        gt_dir_panop, logger, post_proccess_params, visuals_pan_eval, out_dir, metric, evalScale,
                        evaluate_from_saved_numpy_predictions
                                                                                )
                )
            elif eval_type == 'maskrcnn_panoptic_ori_img_shape': # only eval inst seg. mAP
                eval_results.update(
                    super(MapillaryDataset, self).evaluate_panoptic_for_maskrcnn_on_mapillary_ori_img_shapes(
                        results, cuda, panop_eval_temp_folder, dataset_name, gt_dir, debug, num_samples_debug,
                        gt_dir_panop, logger, post_proccess_params, visuals_pan_eval, out_dir
                                                                                )
                )
            else:
                raise NotImplementedError(f'implementation not found for eval_type={eval_type}')

        return eval_results
