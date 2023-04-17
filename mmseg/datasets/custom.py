# -----------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adopted from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Panoptic dataloading and panoptic evaluation on Cityscapes and Mapillary Vistas
# -----------------------------------------------------------------------------------



import os
import os.path as osp
from collections import OrderedDict
from functools import reduce
import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose
from .pipelines import GenPanopLabels, GenPanopLabelsForMaskFormer
import json
from PIL import Image
import time
import torch.nn.functional as F
from tools.panoptic_deeplab.eval import SemanticEvaluator, CityscapesInstanceEvaluator, CityscapesPanopticEvaluator
from tools.panoptic_deeplab.utils import rgb2id
from mmseg.utils.visualize_pred import save_predictions, save_predictions_bottomup
from mmseg.datasets.utils import resize_with_pad
from tools.panoptic_deeplab.post_processing import get_semantic_segmentation, \
      get_panoptic_segmentation, \
    get_cityscapes_instance_format, \
    get_cityscapes_instance_format_for_maskformer, \
    get_cityscapes_instance_format_for_maskrcnn, \
    merge_semantic_and_instance, merge_semantic_and_instance_v2, \
    get_cityscapes_instance_format_for_maskrcnn_v3
from tools.panoptic_deeplab.utils import AverageMeter



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@DATASETS.register_module()
class CustomDataset(Dataset):
    CLASSES = None
    PALETTE = None
    def __init__(self,
                 pipeline,
                 img_dir,
                 depth_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 include_diffusion_data=False,
                 diffusion_set=None,
                 ):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)
        self.include_diffusion_data = include_diffusion_data
        self.diffusion_set = diffusion_set
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)
            if not osp.isabs(self.depth_dir):
                if not self.depth_dir == '':
                    self.depth_dir = osp.join(self.data_root, self.depth_dir)
        # load annotations
        self.img_infos = self.load_annotations_panoptic(self.ann_dir)
        self.gen_panop_labels = GenPanopLabels(8, 'val')
        self.gen_panop_labels_for_maskformer = GenPanopLabelsForMaskFormer(8, 'val', gen_instance_classids_from_zero=True)
        self.best_miou = -1.0

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations_panoptic(self, ann_dir):
        img_infos = []
        if not self.include_diffusion_data:
            json_filename = ann_dir + '.json'
        else:
            json_filename = ann_dir + f'_{self.diffusion_set}.json'
        print_log(f'Loaded annotations from : {json_filename}', logger=get_root_logger())
        dataset = json.load(open(json_filename))
        self.files = {}
        for ano in dataset['annotations']:
            img_info = {}
            if 'synthia' in self.data_root:
                ano_fname = ano['file_name']
                seg_fname = ano['image_id'] + self.seg_map_suffix
            elif 'cityscapes' in self.data_root:
                ano_fname = ano['image_id']
                str1 = ano_fname.split('_')[0] + '/' + ano_fname
                ano_fname = str1 + '_leftImg8bit.png'
                seg_fname = str1 + self.seg_map_suffix
            elif 'mapillary' in self.data_root:
                ano_fname = ano['file_name'].replace('.png', '.jpg')
                seg_fname = ano['file_name']
            img_info['filename'] = ano_fname
            img_info['ann'] = {}
            img_info['ann']['seg_map'] = seg_fname
            img_info['ann']['segments_info'] = ano['segments_info']
            img_infos.append(img_info)
        print_log( f'Loaded {len(img_infos)} images from {self.img_dir}', logger=get_root_logger())
        return img_infos


    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['depth_prefix'] = self.depth_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def get_debug_img_list(self, dataset_name):
        if dataset_name == 'cityscapes':
            img_list = ['frankfurt/frankfurt_000000_000294_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_000576_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_001016_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_001236_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_001751_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_002196_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_002963_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_003025_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_003357_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_003920_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_004617_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_005543_gtFine_panoptic.png']
        elif dataset_name == 'mapillary':
            img_list = [
                '--BJs76vloEaiH-wppzWNA.png',
                '-3-MmXdwhyIQhtb4-8NqHQ.png',
                '-32tlgoydG0ZCyijh8piZQ.png',
                '-4jzRzGfKmQg8RBNlNqnGQ.png',
                '-9y4NjcjdoPFMs5wwC7otg.png',
                '-BYnT4s40fJHAlOumPYbyQ.png',
                '-BqO16ocxK46wM5W-QCE_A.png',
                '-C-x3xSPFIEjqbyVC5PRaQ.png',
                '-DXgAnuaSe6TtQ9Hbm3G2A.png',
                '-F-jXLRFKunhfJg4s-62jA.png',
                '-F5vhdPopdHyJjiC2hI6xg.png',
                '-OB82zvf2k0rTOxuMEuQGA.png',
                '-UqWx1Q0an_GDMMJs3bmOw.png',
            ]
        else:
            raise NotImplementedError(f'No implementation found for datset: {dataset_name}')
        return img_list

    def get_gt_semantic_labels(self):
        """Get ground truth panoptic labels for evaluation."""
        gt_semantic_labels = []
        for img_info in self.img_infos:
            filename = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            panop_lbl_dict = {}
            gt_panoptic_seg = Image.open(filename)
            gt_panoptic_seg = np.asarray(gt_panoptic_seg, dtype=np.float32)  # the id values are > 255, we need np.float32 # (760,1280,3)
            panop_lbl_dict['gt_panoptic_seg'] = gt_panoptic_seg
            panop_lbl_dict['ann_info'] = {}
            panop_lbl_dict['ann_info']['segments_info'] = img_info['ann']['segments_info']
            panop_lbl_dict['seg_fields'] = []
            panop_lbl_dict['seg_fields'].append('gt_panoptic_seg')
            panoptic_labels = self.gen_panop_labels(panop_lbl_dict)
            gt_semantic_labels.append(panoptic_labels['gt_semantic_seg'])
        return gt_semantic_labels

    def get_gt_panoptic_labels(self, device, logger, debug,
                               labels_for_maskformer=False,
                               eval_type=None,
                               dataset_name='cityscapes',
                               evalScale=None,
                               ):
        """Get ground truth panoptic labels for evaluation."""
        img_list = self.get_debug_img_list(dataset_name) if debug else []
        log_interval = 1 if debug else 50
        gt_panoptic_labels = []
        count_img = 0
        logger.info('')
        new_image_shapes = []
        for img_info in self.img_infos:
            if debug and img_info['ann']['seg_map'] not in img_list:
                continue
            filename = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            panop_lbl_dict = {}
            gt_panoptic_seg = Image.open(filename)
            # resize the GT panoptic segmap if required
            if dataset_name == 'cityscapes' and evalScale == '1024x512':
                gt_panoptic_seg = gt_panoptic_seg.resize((1024, 512), Image.NEAREST)
            elif dataset_name == 'cityscapes' and evalScale == '2048x1024':
                pass
            elif dataset_name == 'cityscapes' and evalScale is None:
                pass
            elif dataset_name == 'mapillary':
                gt_panoptic_seg, new_image_shape = resize_with_pad(gt_panoptic_seg, [1024, 768], Image.NEAREST, pad_value=0, is_label=True)
                new_image_shapes.append(new_image_shape)
            else:
                raise NotImplementedError(f'No implementation found for datset: {dataset_name}')
            # convert the PIL image to numpy array
            if dataset_name == 'cityscapes':
                gt_panoptic_seg = np.asarray(gt_panoptic_seg, dtype=np.float32)
            elif dataset_name == 'mapillary':
                pass
            else:
                raise NotImplementedError(f'No implementation found for datset: {dataset_name}')
            panop_lbl_dict['gt_panoptic_seg'] = gt_panoptic_seg
            panop_lbl_dict['ann_info'] = {}
            panop_lbl_dict['ann_info']['segments_info'] = img_info['ann']['segments_info']
            panop_lbl_dict['seg_fields'] = []
            panop_lbl_dict['seg_fields'].append('gt_panoptic_seg')
            if labels_for_maskformer:
                data = self.gen_panop_labels_for_maskformer(panop_lbl_dict)
                gt_panoptic_labels.append([data['gt_semantic_seg'], data['gt_masks'], data['gt_labels'], data['gt_bboxes']])
            else:
                data = self.gen_panop_labels(panop_lbl_dict)
                gt_panoptic_labels.append([data['gt_semantic_seg'], data['gt_center'], data['gt_offset'], data['gt_instance_seg']])
            if count_img % log_interval == 0:
                logger.info(f'generating panoptic labels for imgid:  {count_img+1}')
            count_img+=1
        return gt_panoptic_labels, new_image_shapes

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette


    def evaluate_panoptic_for_maskrcnn( self, results, device=None, panop_eval_temp_folder=None,
                                        dataset_name=None, gt_dir=None, debug=False, num_samples_debug=None,
                                        gt_dir_panop=None, logger=None, post_proccess_params=None,
                                        visuals_pan_eval=None, out_dir=None, metric='mIoU', evalScale=None,
                                        evaluate_from_saved_numpy_predictions=None,
                                        ):
        '''
        This panoptic segmentation evaluation script is adapted from: https://github.com/bowenc0221/panoptic-deeplab
        '''

        if 'mPQ' in metric and 'mAP' not in metric:
            raise AssertionError('mPQ can not be computed with mAP computation.')

        # getting panoptic deeplab post processing params
        num_classes = post_proccess_params['num_classes']
        train_id_to_eval_id = post_proccess_params['train_id_to_eval_id']
        mapillary_dataloading_style = post_proccess_params['mapillary_dataloading_style']
        ignore_label = post_proccess_params['ignore_label']
        label_divisor = post_proccess_params['label_divisor']
        thing_list = post_proccess_params['thing_list']
        mask_th = post_proccess_params['mask_score_threshold']
        use_semantic_decoder_for_instance_labeling = post_proccess_params['use_semantic_decoder_for_instance_labeling']
        use_semantic_decoder_for_panoptic_labeling = post_proccess_params['use_semantic_decoder_for_panoptic_labeling']
        nms_th = post_proccess_params['nms_th']
        intersec_th = post_proccess_params['intersec_th']
        upsnet_mask_pruning = post_proccess_params['upsnet_mask_pruning']
        generate_thing_cls_panoptic_from_instance_pred = post_proccess_params['generate_thing_cls_panoptic_from_instance_pred']

        # The below params are to evaluate the M-Net, i.e., you first train two networks for semseg and instseg.
        # Once training is done, we fuse the predictions of these two networks at inference time.
        # to generate the panoptic segmentation prediction maps, and evaluate the predictions to get a mPQ for M-Net .
        # (c.f. Please refer to our main paper for M-Net)
        DUMP_SEMANTIC_PRED_AS_NUMPY_ARRAY = post_proccess_params['dump_semantic_pred_as_numpy_array']
        LOAD_SEMANTIC_PRED_AS_NUMPY_ARRAY = post_proccess_params['load_semantic_pred_as_numpy_array']
        semantic_pred_numpy_array_location = post_proccess_params['semantic_pred_numpy_array_location']

        if upsnet_mask_pruning:
            assert nms_th
            assert intersec_th

        # get all the GT panoptic labels for all images in the val set
        gt_panoptic_labels, new_img_shapes = self.get_gt_panoptic_labels(
                                                                            device,
                                                                            logger,
                                                                            debug,
                                                                            labels_for_maskformer=True,
                                                                            dataset_name=dataset_name,
                                                                            evalScale=evalScale
                                                                        )

        # debug time setting
        log_interval = 1 if debug else 50
        num_samples = num_samples_debug if debug else len(gt_panoptic_labels)
        numpys_path = None
        if evaluate_from_saved_numpy_predictions:
            strs = panop_eval_temp_folder.split('/')
            numpys_path = os.path.join(strs[0], strs[1], strs[2], strs[3], strs[4], 'results_numpys')
            npy_file_list = os.listdir(numpys_path)
            assert len(gt_panoptic_labels) == len(npy_file_list), 'The number of gt labels and predictions are not the same !!'
            pass
        else:
            assert len(gt_panoptic_labels) == len(results), 'The number of gt labels and predictions are not the same !!'

        # creating folders to dump PNGs generated during panoptc-deeplab evaluation
        eval_folder = {}
        eval_folder['instance'] = os.path.join(panop_eval_temp_folder, 'instance')
        eval_folder['visuals'] = os.path.join(panop_eval_temp_folder, 'visuals')
        eval_folder['semantic'] = os.path.join(panop_eval_temp_folder, 'semantic')
        eval_folder['panoptic'] = os.path.join(panop_eval_temp_folder, 'panoptic')

        # setting JSON files
        image_filename_list = []
        for i in range(len(self.img_infos)):
            image_filename_list.append(self.img_infos[i]['ann']['seg_map'].split('.')[0])

        if dataset_name == 'cityscapes':
            panoptic_josn_file = 'cityscapes_panoptic_val.json'
            panoptic_json_folder = 'cityscapes_panoptic_val'
            stuff_area = 2048
            input_image_size = (1024, 512) if debug else (2048, 1024) # Not in use
        elif dataset_name == 'mapillary':
            panoptic_josn_file = 'val_panoptic_19cls.json'
            panoptic_json_folder = 'val_panoptic_19cls'
            stuff_area = 2048
            input_image_size = new_img_shapes # (1024, 768) # Not in use
            assert new_img_shapes, 'new_img_shape must not be None'
        else:
            raise NotImplementedError(f'Implementation not found for dataset: {dataset_name}')

        post_time = AverageMeter()
        timing_warmup_iter = 10

        # Initialzing the metrics class objects
        instance_metric = CityscapesInstanceEvaluator(
            output_dir=eval_folder['instance'],
            train_id_to_eval_id=train_id_to_eval_id,
            gt_dir=gt_dir,
            num_classes=num_classes,
            DEBUG=debug,
            num_samples=num_samples_debug,
            dataset_name=dataset_name,
            rgb2id=rgb2id,
            input_image_size=input_image_size,  # Not in use
            mapillary_dataloading_style=mapillary_dataloading_style, # Not in use
            logger=logger,
        )
        semantic_metric = SemanticEvaluator(
            num_classes=num_classes,
            ignore_label=ignore_label,
            output_dir=eval_folder['semantic'],
            train_id_to_eval_id=train_id_to_eval_id,
            logger=logger,
            dataset_name=dataset_name,
        )
        panoptic_metric = CityscapesPanopticEvaluator(
            output_dir=eval_folder['panoptic'],
            train_id_to_eval_id=train_id_to_eval_id,
            label_divisor=label_divisor,
            void_label=label_divisor * ignore_label,
            gt_dir=gt_dir_panop,
            split='val',
            num_classes=num_classes,
            panoptic_josn_file=panoptic_josn_file,
            panoptic_json_folder=panoptic_json_folder,
            debug=debug,
            target_dataset_name=dataset_name,
            input_image_size=input_image_size, # Not in use
            mapillary_dataloading_style=mapillary_dataloading_style, # Not in use
            logger=logger,
        )

        image_filename_list_debug = []
        try:
            for i in range(num_samples):
                if dataset_name == 'cityscapes':
                    image_filename = image_filename_list[i].split('/')[1]
                elif dataset_name == 'mapillary':
                    image_filename = image_filename_list[i]
                else:
                    raise NotImplementedError(f'Implementation not found for dataset: {dataset_name}')
                if i == timing_warmup_iter:
                    post_time.reset()
                start_time = time.time()
                out_dict = {}
                gt_labels = {}

                if evaluate_from_saved_numpy_predictions:
                    numpy_file_path = os.path.join(numpys_path, f'{image_filename}.npy')
                    result_npy = np.load(numpy_file_path, allow_pickle=True)
                    # instance seg
                    out_dict['boxes'] = result_npy[0]['ins_results'][0][0]
                    out_dict['masks'] = result_npy[0]['ins_results'][0][1]
                    # semantic seg
                    out_dict['semantic'] = result_npy[0]['sem_results'][0]
                else:
                    # instance seg
                    out_dict['boxes'] = results[i]['ins_results'][0][0]
                    out_dict['masks'] = results[i]['ins_results'][0][1]
                    # semantic seg
                    out_dict['semantic'] = results[i]['sem_results'][0]

                gt_labels['semantic'] = gt_panoptic_labels[i][0]

                if 'mIoU' in metric and 'mAP' not in metric and 'mPQ' not in metric and evalScale == '1024x512':
                    semantic_prediction = torch.from_numpy(results[i]['sem_results'][0]).float().to(device)
                    semantic_prediction = semantic_prediction.unsqueeze(dim=0)
                    semantic_prediction = semantic_prediction.unsqueeze(dim=0)
                    semantic_prediction = F.interpolate(semantic_prediction, size=(512, 1024), mode='bilinear', align_corners=False)
                    semantic_prediction = semantic_prediction.long()
                    semantic_prediction = semantic_prediction.squeeze(dim=0)
                    semantic_prediction = semantic_prediction.squeeze(dim=0)
                    out_dict['semantic'] = semantic_prediction.cpu().numpy()

                pred_shape = out_dict['semantic'].shape

                if LOAD_SEMANTIC_PRED_AS_NUMPY_ARRAY:
                    assert not DUMP_SEMANTIC_PRED_AS_NUMPY_ARRAY, 'DUMP_SEMANTIC_PRED_AS_NUMPY_ARRAY and LOAD_SEMANTIC_PRED_AS_NUMPY_ARRAY can not be True at the same time!'
                    assert semantic_pred_numpy_array_location is not None, 'if LOAD_SEMANTIC_PRED_AS_NUMPY_ARRAY is True then you need to provide ' \
                                                                           'the semantic_pred_numpy_array_location as string path in experiments.py'
                    cityname = image_filename_list[i].split('/')[0]
                    npload_file_path1 = os.path.join(semantic_pred_numpy_array_location, 'mnet_exp', cityname)
                    npload_file_path = os.path.join(npload_file_path1, f'{image_filename}.npy')
                    with open(npload_file_path, 'rb') as f:
                        out_dict['semantic'] = np.load(f)
                    logger.info(f'Semantic predictions loaded from: {npload_file_path}')

                if  DUMP_SEMANTIC_PRED_AS_NUMPY_ARRAY:
                    assert not LOAD_SEMANTIC_PRED_AS_NUMPY_ARRAY, 'DUMP_SEMANTIC_PRED_AS_NUMPY_ARRAY and LOAD_SEMANTIC_PRED_AS_NUMPY_ARRAY can not be True at the same time'
                    # dump the semantic pred
                    cityname = image_filename_list[i].split('/')[0]
                    npsave_file_path1 = os.path.join(eval_folder['semantic'], 'mnet_exp', cityname)
                    os.makedirs(npsave_file_path1, exist_ok=True)
                    npsave_file_path = os.path.join(npsave_file_path1, f'{image_filename}.npy')
                    np.save(npsave_file_path, out_dict['semantic'])
                    logger.info(f'Semantic predictions saved at: {npsave_file_path}')

                # instance_metric update
                if 'mAP' in metric:
                    if upsnet_mask_pruning:
                        instances, ins_seg, pan_seg_thing = get_cityscapes_instance_format_for_maskrcnn_v3(
                                                                            out_dict['boxes'],
                                                                            out_dict['masks'],
                                                                            pred_shape=pred_shape,
                                                                            mask_score_th=mask_th,
                                                                            sem_seg=out_dict['semantic'],
                                                                            device=device,
                                                                            thing_list=thing_list,
                                                                            use_semantic_decoder_for_instance_labeling=use_semantic_decoder_for_instance_labeling,
                                                                            use_semantic_decoder_for_panoptic_labeling=use_semantic_decoder_for_panoptic_labeling,
                                                                            nms_th=nms_th,
                                                                            intersec_th=intersec_th,
                                                                                                        )
                    else:
                        instances, ins_seg, pan_seg_thing = get_cityscapes_instance_format_for_maskrcnn(
                                                                            out_dict['boxes'],
                                                                            out_dict['masks'],
                                                                            pred_shape=pred_shape,
                                                                            mask_score_th=mask_th,
                                                                            sem_seg=out_dict['semantic'],
                                                                            device=device,
                                                                            thing_list=thing_list,
                                                                            use_semantic_decoder_for_instance_labeling=use_semantic_decoder_for_instance_labeling,
                                                                            use_semantic_decoder_for_panoptic_labeling=use_semantic_decoder_for_panoptic_labeling,
                                                                            nms_th=nms_th,
                                                                            intersec_th=intersec_th,
                                                                        )

                    instance_metric.update(instances, image_filename, debug=False, logger=logger)

                if 'mIoU' in metric:
                    # semanitc metric update
                    semantic_metric.update(out_dict['semantic'], gt_labels['semantic'], image_filename, debug=debug, logger=logger)

                if 'mPQ' in metric:
                    # generatig the panoptic segmentation from semantic and instance segs
                    out_dict['semantic'] = torch.from_numpy(out_dict['semantic']).long().to(device)
                    ins_seg = torch.from_numpy(ins_seg).long().to(device)

                    if generate_thing_cls_panoptic_from_instance_pred and pan_seg_thing is not None:
                        pan_seg_thing = torch.from_numpy(pan_seg_thing).long().to(device)
                        panoptic_pred = merge_semantic_and_instance_v2(out_dict['semantic'].unsqueeze(dim=0),  # [1, 512, 1024]
                                                                    pan_seg_thing.unsqueeze(dim=0),  # [1, 512, 1024]
                                                                    label_divisor,
                                                                    thing_list,
                                                                    stuff_area,
                                                                    void_label=label_divisor * ignore_label)  # 255000
                    else:
                        panoptic_pred = merge_semantic_and_instance(out_dict['semantic'].unsqueeze(dim=0),  # [1, 512, 1024]
                                                                    ins_seg.unsqueeze(dim=0),  # [1, 512, 1024]
                                                                    label_divisor,
                                                                    thing_list,
                                                                    stuff_area,
                                                                    void_label=label_divisor * ignore_label)  # 255000


                    # panoptic_metric update
                    if 'cityscapes' in dataset_name:
                        image_id = '_'.join(image_filename.split('_')[:3])
                    elif 'mapillary' in dataset_name:
                        image_id = image_filename

                    if 'mPQ' in metric:
                        panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()
                        panoptic_metric.update(panoptic_pred, image_filename=image_filename, image_id=image_id, debug=debug, logger=logger)

                image_filename_list_debug.append(image_filename)

                # Logging
                torch.cuda.synchronize(device)
                post_time.update(time.time() - start_time)
                if i % log_interval == 0:
                    logger.info('[{}/{}]\tPost-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(i, num_samples, post_time=post_time))

                if visuals_pan_eval:
                    TrgSemGT = gt_labels['semantic']
                    TrgSemPd = out_dict['semantic'].cpu().numpy()
                    TrgPanPd = panoptic_pred
                    outdir = eval_folder['visuals']
                    save_predictions(dataset_name, image_filename, TrgSemGT, TrgSemPd, TrgPanPd, outdir, debug, self.img_dir, self.ann_dir, resize_with_pad)

        except Exception:
            logger.exception("Exception during testing:")
            raise
        finally:
            eval_results = {}
            logger.info("Inference finished.")
            logger.info("Evaluating ...")
            if 'mAP' in metric:
                instance_results = instance_metric.evaluate(img_list_debug=image_filename_list_debug)
                logger.info(instance_results)
                eval_results['instance_eval'] = instance_results
            if 'mPQ' in metric:
                panoptic_results = panoptic_metric.evaluate(logger)
                logger.info(panoptic_results)
                mPQ = panoptic_results['All']['pq']
                eval_results['panoptic_eval'] = panoptic_results
            if 'mIoU' in metric:
                semantic_results = semantic_metric.evaluate()
                logger.info(semantic_results)
                mIoU = semantic_results['sem_seg']['mIoU']
                eval_results['semantic_eval'] = semantic_results
            logger.info('END: panoptic evaluation !')
            if 'mIoU' in metric:
                if self.best_miou < mIoU:
                    self.best_miou = mIoU
                    logger.info('*** BEST mIoU: {} ***'.format(self.best_miou))
            if 'mPQ' in metric:
                logger.info('*** Corresponding PQ: {} ***'.format(mPQ))
            # removing the intermediate results and keeping the final evaluation results (json files)
            if 'mAP' in metric:
                strCmd1 = 'rm -r' + ' ' + eval_folder['instance']
                os.system(strCmd1)
                logger.info(f'executing : {strCmd1}')
            if 'mIoU' in metric and not DUMP_SEMANTIC_PRED_AS_NUMPY_ARRAY:
                strCmd2 = 'rm -r' + ' ' + eval_folder['semantic']
                os.system(strCmd2)
                logger.info(f'executing : {strCmd2}')
            if 'mPQ' in metric:
                strCmd3 = 'rm -r' + ' ' + os.path.join(eval_folder['panoptic'], 'predictions')
                os.system(strCmd3)
                logger.info(f'executing : {strCmd3}')
            logger.info('mask_score_th:')
            logger.info(mask_th)
            logger.info('Removing the intermediate results and keeping the final eval json files ...')
            return eval_results


    def evaluate_panoptic(self, results, device=None, panop_eval_temp_folder=None,
                          dataset_name=None, gt_dir=None, debug=False, num_samples_debug=None,
                          gt_dir_panop=None, logger=None, post_proccess_params=None,
                          visuals_pan_eval=None, evalScale=None, metric='mIoU'):
        gt_panoptic_labels, new_image_shapes = \
                                                self.get_gt_panoptic_labels(
                                                    device, logger, debug,
                                                    labels_for_maskformer=False,
                                                    dataset_name=dataset_name,
                                                    evalScale=evalScale,
                                                )
        if 'mPQ' in metric and 'mAP' not in metric:
            raise AssertionError('mPQ can not be computed with mAP computation.')
        if debug:
            log_interval=1
            num_samples = num_samples_debug
        else:
            log_interval = 50
            num_samples = len(gt_panoptic_labels)
        eval_folder = {}
        eval_folder['semantic'] = os.path.join(panop_eval_temp_folder, 'semantic')
        eval_folder['instance'] = os.path.join(panop_eval_temp_folder, 'instance')
        eval_folder['panoptic'] = os.path.join(panop_eval_temp_folder, 'panoptic')
        eval_folder['visuals'] = os.path.join(panop_eval_temp_folder, 'visuals')
        image_filename_list = []
        for i in range(len(self.img_infos)):
            image_filename_list.append(self.img_infos[i]['ann']['seg_map'].split('.')[0])
        if dataset_name == 'cityscapes':
            panoptic_josn_file = 'cityscapes_panoptic_val.json'
            panoptic_json_folder = 'cityscapes_panoptic_val'
            stuff_area = 2048
            input_image_size = (2048, 1024)
        elif dataset_name == 'mapillary':
            panoptic_josn_file = 'val_panoptic_19cls_1024x768.json'
            panoptic_json_folder = 'val_panoptic_19cls_1024x768'
            stuff_area = 2048
            input_image_size = (1024, 768)
        else:
            raise NotImplementedError(f'Implementation not found for dataset: {dataset_name}')
        num_classes = post_proccess_params['num_classes']
        ignore_label = post_proccess_params['ignore_label']
        train_id_to_eval_id = post_proccess_params['train_id_to_eval_id']
        mapillary_dataloading_style = post_proccess_params['mapillary_dataloading_style']
        label_divisor = post_proccess_params['label_divisor']
        cityscapes_thing_list = post_proccess_params['cityscapes_thing_list']
        CENTER_THRESHOLD = 0.1 # post_proccess_params['center_threshold']
        NMS_KERNEL = post_proccess_params['nms_kernel']
        TOP_K_INSTANCE = post_proccess_params['top_k_instance']
        post_time = AverageMeter()
        timing_warmup_iter = 10
        INSTANCE_SCORE_TYPE = 'semantic'
        semantic_metric = SemanticEvaluator(
            num_classes=num_classes,
            ignore_label=ignore_label,
            output_dir=eval_folder['semantic'],
            train_id_to_eval_id=train_id_to_eval_id,
            logger=logger,
            dataset_name=dataset_name,
        )
        instance_metric = CityscapesInstanceEvaluator(
            output_dir=eval_folder['instance'],
            train_id_to_eval_id=train_id_to_eval_id,
            gt_dir=gt_dir,
            num_classes=num_classes,
            DEBUG=debug,
            num_samples=num_samples_debug,
            dataset_name=dataset_name,
            rgb2id=rgb2id,
            input_image_size=input_image_size,
            mapillary_dataloading_style=mapillary_dataloading_style,
            logger=logger,
        )
        panoptic_metric = CityscapesPanopticEvaluator(
            output_dir=eval_folder['panoptic'],
            train_id_to_eval_id=train_id_to_eval_id,
            label_divisor=label_divisor,
            void_label=label_divisor * ignore_label,
            gt_dir=gt_dir_panop,
            split='val',
            num_classes=num_classes,
            panoptic_josn_file=panoptic_josn_file,
            panoptic_json_folder=panoptic_json_folder,
            debug=debug,
            target_dataset_name=dataset_name,
            input_image_size=input_image_size,
            mapillary_dataloading_style=mapillary_dataloading_style,
            logger=logger,
        )
        image_filename_list_debug = []
        try:
            for i in range(num_samples):
                if dataset_name == 'cityscapes':
                    image_filename = image_filename_list[i].split('/')[1]
                elif dataset_name == 'mapillary':
                    image_filename = image_filename_list[i]
                else:
                    raise NotImplementedError(f'Implementation not found for dataset: {dataset_name}')
                if i == timing_warmup_iter:
                    post_time.reset()
                start_time = time.time()
                out_dict = {}
                out_dict['semantic'] = torch.from_numpy(results[i]['semantic']).to(device)
                out_dict['center'] = torch.from_numpy(results[i]['center']).to(device)
                out_dict['offset'] = torch.from_numpy(results[i]['offset']).to(device)
                if evalScale == '2048x1024' or evalScale is None:
                    out_dict['semantic'] = F.interpolate(out_dict['semantic'], size=(1024, 2048), mode='bilinear', align_corners=False)
                    out_dict['center'] = F.interpolate(out_dict['center'], size=(1024, 2048), mode='bilinear', align_corners=False)
                    out_dict['offset'] = F.interpolate(out_dict['offset'], size=(1024, 2048), mode='bilinear', align_corners=False)
                elif evalScale == '1024x512':
                    out_dict['semantic'] = F.interpolate(out_dict['semantic'], size=(512, 1024), mode='bilinear', align_corners=False)
                    out_dict['center'] = F.interpolate(out_dict['center'], size=(512, 1024), mode='bilinear', align_corners=False)
                    out_dict['offset'] = F.interpolate(out_dict['offset'], size=(512, 1024), mode='bilinear', align_corners=False)
                else:
                    raise NotImplementedError(f'No implementation found for evalScale:{evalScale}')
                gt_labels = {}
                gt_labels['semantic'] = torch.from_numpy(gt_panoptic_labels[i][0]).to(device)
                # the following three gt labels are not used by the evaluation script, they are just used for visualization purpose
                gt_labels['center'] = gt_panoptic_labels[i][1]
                gt_labels['offset'] = gt_panoptic_labels[i][2]
                gt_labels['gt_instance_seg'] = gt_panoptic_labels[i][3]
                semantic_pred = get_semantic_segmentation(out_dict['semantic'])
                if 'mPQ' in metric:
                    panoptic_pred, center_pred = get_panoptic_segmentation(
                        semantic_pred,
                        out_dict['center'],
                        out_dict['offset'],
                        thing_list=cityscapes_thing_list,
                        label_divisor=label_divisor,
                        stuff_area=stuff_area,
                        void_label=(label_divisor * ignore_label),
                        threshold=CENTER_THRESHOLD,
                        nms_kernel=NMS_KERNEL,
                        top_k=TOP_K_INSTANCE,
                        foreground_mask=None)
                    panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()
                torch.cuda.synchronize(device)
                post_time.update(time.time() - start_time)
                if i % log_interval == 0:
                    logger.info('[{}/{}]\tPost-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(i, num_samples, post_time=post_time))
                semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                # Evaluates semantic segmentation.
                if 'mIoU' in metric:
                    semantic_metric.update(semantic_pred, gt_labels['semantic'].squeeze(0).cpu().numpy(), image_filename, debug=debug, logger=logger)
                # Evaluates instance segmentation.
                raw_semantic = F.softmax(out_dict['semantic'], dim=1)
                center_hmp = out_dict['center']
                raw_semantic = raw_semantic.squeeze(0).cpu().numpy()
                center_hmp = center_hmp.squeeze(1).squeeze(0).cpu().numpy()
                if 'mAP' in metric:
                    instances = get_cityscapes_instance_format(panoptic_pred, raw_semantic, center_hmp, label_divisor=label_divisor, score_type=INSTANCE_SCORE_TYPE)
                    instance_metric.update(instances, image_filename, debug=debug, logger=logger)
                # Evaluates panoptic segmentation.
                if 'cityscapes' in dataset_name:
                    image_id = '_'.join(image_filename.split('_')[:3])
                elif 'mapillary' in dataset_name:
                    image_id = image_filename
                if 'mPQ' in metric:
                    panoptic_metric.update(panoptic_pred, image_filename=image_filename, image_id=image_id)
                image_filename_list_debug.append(image_filename)
                if visuals_pan_eval:
                    save_predictions_bottomup(gt_labels['semantic'].cpu().numpy(),
                                            gt_labels['center'],
                                            gt_labels['offset'],
                                            gt_labels['gt_instance_seg'],
                                            eval_folder['visuals'],
                                            image_filename_list[i],
                                            semantic_pred,
                                            center_hmp,
                                            out_dict['offset'],
                                            panoptic_pred,
                                            debug,
                                            dataset_name,
                                            logger
                                            )
        except Exception:
            logger.exception("Exception during testing:")
            raise
        finally:
            eval_results = {}
            logger.info("Inference finished.")
            if 'mIoU' in metric:
                semantic_results = semantic_metric.evaluate()
                logger.info(semantic_results)
                eval_results['semantic_eval'] = semantic_results
                mIoU = semantic_results['sem_seg']['mIoU']
            if instance_metric is not None and 'mAP' in metric:
                instance_results = instance_metric.evaluate(img_list_debug=image_filename_list_debug)
                logger.info(instance_results)
                eval_results['instance_eval'] = instance_results
            if panoptic_metric is not None and 'mPQ' in metric:
                panoptic_results = panoptic_metric.evaluate(logger)
                logger.info(panoptic_results)
                eval_results['panoptic_eval'] = panoptic_results
                mPQ = panoptic_results['All']['pq']
            if self.best_miou < mIoU:
                self.best_miou = mIoU
                logger.info('*** BEST mIoU: {} ***'.format(self.best_miou))
                if panoptic_metric is not None and 'mPQ' in metric:
                    logger.info('*** Corresponding PQ: {} ***'.format(mPQ))
            # removing the intermediate results and keeping the final evaluation results (json files)
            strCmd1 = 'rm -r' + ' ' + eval_folder['instance']
            strCmd2 = 'rm -r' + ' ' + eval_folder['semantic']
            strCmd3 = 'rm -r' + ' ' + os.path.join(eval_folder['panoptic'], 'predictions')
            logger.info('Removing the intermediate results and keeping the final eval json files ...')
            if 'mAP' in metric:
                os.system(strCmd1)
            if 'mIoU' in metric:
                os.system(strCmd2)
            if 'mPQ' in metric:
                os.system(strCmd3)
            logger.info('END: panoptic evaluation !')
            logger.info('')
            return eval_results