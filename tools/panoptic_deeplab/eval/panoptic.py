# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/panoptic_evaluation.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import contextlib
import io
import logging
from collections import OrderedDict
import os
import json

import numpy as np

# from fvcore.common.file_io import PathManager
# from ctrl.utils.panoptic_deeplab import save_annotation

from fvcore.common.file_io import PathManager
from tools.panoptic_deeplab.save_annotations import save_annotation


class CityscapesPanopticEvaluator:
    """
    Evaluate panoptic segmentation
    """
    def __init__(
                    self, output_dir=None,
                    train_id_to_eval_id=None,
                    label_divisor=1000,
                    void_label=255000,
                    gt_dir='./datasets/cityscapes',
                    split='val',
                    num_classes=19,
                    panoptic_josn_file=None,
                    panoptic_json_folder=None,
                    debug=None,
                    target_dataset_name=None,
                    input_image_size=None,
                    mapillary_dataloading_style='OURS',
                    logger=None
                ):
        """
        Args:
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
            label_divisor (int):
            void_label (int):
            gt_dir (str): path to ground truth annotations.
            split (str): evaluation split.
            num_classes (int): number of classes.
        """
        self.debug = debug
        if output_dir is None:
            raise ValueError('Must provide a output directory.')
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._panoptic_dir = os.path.join(self._output_dir, 'predictions')
        if self._panoptic_dir:
            PathManager.mkdirs(self._panoptic_dir)

        self._predictions = []
        self._predictions_json = os.path.join(output_dir, 'predictions.json')

        self._train_id_to_eval_id = train_id_to_eval_id
        self._label_divisor = label_divisor
        self._void_label = void_label
        self._num_classes = num_classes
        self.dataset_name = target_dataset_name
        self.input_image_size = input_image_size
        self.mapillary_dataloading_style = mapillary_dataloading_style

        self._logger = logger

        self._logger.info('tools/panoptic_deeplab/eval/panoptic.py --> class CityscapesPanopticEvaluator: --> def __init__() --> self._logger : {}'.format(self._logger))

        self._gt_json_file = os.path.join(gt_dir, panoptic_josn_file)
        self._gt_folder = os.path.join(gt_dir, panoptic_json_folder)

        # if 'cityscapes' in target_dataset_name:
        #     self._gt_json_file = os.path.join(gt_dir, panoptic_josn_file)
        #     self._gt_folder = os.path.join(gt_dir, panoptic_json_folder)
        #
        # elif 'mapillary' in target_dataset_name:
        #     self._gt_json_file = os.path.join(gt_dir, panoptic_josn_file)
        #     self._gt_folder = os.path.join(gt_dir, panoptic_json_folder)
        # else:
        #     NotImplementedError('no implmentation error --> ctrl/eval_panop/panoptic.py --> def __init__() --> class CityscapesPanopticEvaluator()')

        self._pred_json_file = os.path.join(output_dir, 'predictions.json')
        self._pred_folder = self._panoptic_dir
        self._resultsFile = os.path.join(output_dir, 'resultPanopticSemanticLabeling.json')

    @staticmethod
    def id2rgb(id_map):
        if isinstance(id_map, np.ndarray):
            id_map_copy = id_map.copy()
            rgb_shape = tuple(list(id_map.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            for i in range(3):
                rgb_map[..., i] = id_map_copy % 256
                id_map_copy //= 256
            return rgb_map
        color = []
        for _ in range(3):
            color.append(id_map % 256)
            id_map //= 256
        return color

    def update(self, panoptic, image_filename=None, image_id=None, debug=False, logger=None):
        if image_filename is None:
            raise ValueError('Need to provide image_filename.')
        if image_id is None:
            raise ValueError('Need to provide image_id.')

        # Change void region.
        panoptic[panoptic == self._void_label] = 0

        segments_info = []
        for pan_lab in np.unique(panoptic):
            pred_class = pan_lab // self._label_divisor
            if self._train_id_to_eval_id is not None:
                pred_class = self._train_id_to_eval_id[pred_class]
            segments_info.append(
                {
                    'id': int(pan_lab),
                    'category_id': int(pred_class),
                }
            )
        save_annotation(self.id2rgb(panoptic), self._panoptic_dir, image_filename, add_colormap=False, debug=debug, logger=logger)
        self._predictions.append(
            {
                'image_id': image_id,
                'file_name': image_filename + '.png',
                'segments_info': segments_info,
            }
        )

    def evaluate(self, logger):
        import cityscapesscripts.evaluation.evalPanopticSemanticLabeling as cityscapes_eval

        gt_json_file = self._gt_json_file
        gt_folder = self._gt_folder
        pred_json_file = self._pred_json_file
        pred_folder = self._pred_folder
        resultsFile = self._resultsFile

        with open(gt_json_file, "r") as f:
            json_data = json.load(f)
        json_data["annotations"] = self._predictions
        with PathManager.open(self._predictions_json, "w") as f:
            f.write(json.dumps(json_data))

        with contextlib.redirect_stdout(io.StringIO()):
            results = cityscapes_eval.evaluatePanoptic(
                                                        gt_json_file,
                                                        gt_folder,
                                                        pred_json_file,
                                                        pred_folder,
                                                        resultsFile,
                                                        debug=self.debug,
                                                        dataset_name=self.dataset_name,
                                                        input_image_size=self.input_image_size,
                                                        mapillary_dataloading_style=self.mapillary_dataloading_style,
                                                        logger=logger
                                                    )

        self._logger.info(results)
        return results
