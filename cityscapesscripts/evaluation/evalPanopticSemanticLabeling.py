#!/usr/bin/python
#
# The evaluation script for panoptic segmentation (https://arxiv.org/abs/1801.00868).
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
# Test set evaluation assumes prediction use 'id' and not 'trainId'
# for categories, i.e. 'person' id is 24.
#
# The script expects both ground truth and predictions to use COCO panoptic
# segmentation format (http://cocodataset.org/#format-data and
# http://cocodataset.org/#format-results respectively). The format has 'image_id' field to
# match prediction and annotation. For cityscapes we assume that the 'image_id' has form
# <city>_123456_123456 and corresponds to the prefix of cityscapes image files.
#
# Note, that panoptic segmentaion in COCO format is not included in the basic dataset distribution.
# To obtain ground truth in this format, please run script 'preparation/createPanopticImgs.py'
# from this repo. The script is quite slow and it may take up to 5 minutes to convert val set.
#

# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import sys
import argparse
import functools
import traceback
import json
import time
import multiprocessing
import numpy as np
from collections import defaultdict

# Image processing
from PIL import Image

# Cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
# from cityscapesscripts.helpers.labels import labels as csLabels
# from ctrl.dataset.mapillary_panop_jan04_2022_v2 import pad_with_fixed_AS, resize_with_pad
from mmseg.datasets.utils import resize_with_pad, pad_with_fixed_AS

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

OFFSET = 256 * 256 * 256
VOID = 0



# The decorator is used to prints an error trhown inside process
def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in worker thread:')
            traceback.print_exc()
            raise e

    return wrapper


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories,
                           dataset_name, input_image_size, mapillary_dataloading_style, logger, debug):
    pq_stat = PQStat()
    # idx = 0
    # str1 = []
    for gt_ann, pred_ann in annotation_set:
        # if idx % 30 == 0:
        #     logger.info('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        # idx += 1

        if 'mapillary' in dataset_name:
            pan_gt = Image.open(os.path.join(gt_folder, gt_ann['file_name']))
            if mapillary_dataloading_style == 'DADA':
                raise NotImplementedError('To evaluate the mapillary on original image shape for panoptic seg,'
                                          ' you need to first upsample the predicted masks with pad_with_fixed_AS(). '
                                          'This part is not implemented yet.')
                # target_ratio = 1024 / 768
                # pan_gt = pad_with_fixed_AS(target_ratio, pan_gt, fill_value=0, is_label=False)
            else:
                pan_gt, new_image_shape = resize_with_pad(pan_gt, [1024, 768], Image.NEAREST, pad_value=0, is_label=True)
            pan_gt = rgb2id(pan_gt).astype(np.int32)

        elif 'cityscapes' in dataset_name:
            sub_folder_name = gt_ann['file_name'].split('_')[0]
            pan_gt = Image.open(os.path.join(gt_folder, sub_folder_name, gt_ann['file_name']))
            if debug:
                pan_gt = pan_gt.resize((1024, 512), Image.NEAREST)
            pan_gt = np.array(pan_gt, dtype=np.uint32)
            # pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32) # original
            pan_gt = rgb2id(pan_gt)
        else:
            NotImplementedError('no implementation found at cityscapesscripts/evaluation/evalPanopticSemanticLabeling.py --> def pq_compute_single_core(...)')

        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union

            # str1.append('pred_label:{}, pred_area:{}, gt_label:{}, gt_area:{}, intsec:{}, gt_pred_map.get((VOID, pred_label):{}, union: {}, iou:{}'.
            #             format(pred_label, pred_segms[pred_label]['area'], gt_label, gt_segms[gt_label]['area'], intersection, gt_pred_map.get((VOID, pred_label), 0), union, iou)
            #             )

            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false negative
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    # logger.info('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, debug,
                          dataset_name, input_image_size=None, mapillary_dataloading_style='OURS', logger=None):

    if debug:
        # def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
        cpu_num = 1
        annotations_split = np.array_split(matched_annotations_list, cpu_num)
        for proc_id, annotation_set in enumerate(annotations_split):
            pq_stat = pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder,
                                             categories, dataset_name, input_image_size,
                                             mapillary_dataloading_style, logger, debug)
    else:
        cpu_num = multiprocessing.cpu_count()
        annotations_split = np.array_split(matched_annotations_list, cpu_num)
        # logger.info("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
        workers = multiprocessing.Pool(processes=cpu_num)
        processes = []
        for proc_id, annotation_set in enumerate(annotations_split):
            p = workers.apply_async(pq_compute_single_core, (proc_id, annotation_set, gt_folder, pred_folder,
                                                             categories, dataset_name, input_image_size,
                                                             mapillary_dataloading_style, logger, debug))
            processes.append(p)
        pq_stat = PQStat()
        for p in processes:
            pq_stat += p.get()
        workers.close()

    # for line in print_log:
    #     logger.info(line)

    return pq_stat




def average_pq(pq_stat, categories):
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    return results


def print_results(results, categories, logger):
    metrics = ["All", "Things", "Stuff"]
    logger.info("{:14s}| {:>5s}  {:>5s}  {:>5s}".format("Category", "PQ", "SQ", "RQ"))
    labels = sorted(results['per_class'].keys())
    for label in labels:
        logger.info("{:14s}| {:5.1f}  {:5.1f}  {:5.1f}".format(
            categories[label]['name'],
            100 * results['per_class'][label]['pq'],
            100 * results['per_class'][label]['sq'],
            100 * results['per_class'][label]['rq']
        ))
    logger.info("-" * 41)
    logger.info("{:14s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))

    for name in metrics:
        logger.info("{:14s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n']
        ))


def evaluatePanoptic(
                        gt_json_file,
                        gt_folder,
                        pred_json_file,
                        pred_folder,
                        resultsFile,
                        debug=None,
                        dataset_name=None,
                        input_image_size=None,
                        mapillary_dataloading_style='OURS',
                        logger=None
                        ):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)
    categories = {el['id']: el for el in gt_json['categories']}

    # if debug:
    # import logging
    # _logger = logging.getLogger(__name__)
    _logger = logger

    _logger.info("Evaluation panoptic segmentation metrics:")
    _logger.info("Ground truth:")
    _logger.info("\tSegmentation folder: {}".format(gt_folder))
    _logger.info("\tJSON file: {}".format(gt_json_file))
    _logger.info("Prediction:")
    _logger.info("\tSegmentation folder: {}".format(pred_folder))
    _logger.info("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        printError("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        printError("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if debug:
            if image_id in pred_annotations:
                matched_annotations_list.append((gt_ann, pred_annotations[image_id]))
        else:
            if image_id not in pred_annotations:
                raise Exception('no prediction for the image with id: {}'.format(image_id))
            matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(
                                        matched_annotations_list,
                                        gt_folder,
                                        pred_folder,
                                        categories,
                                        debug,
                                        dataset_name,
                                        input_image_size,
                                        mapillary_dataloading_style,
                                        logger
                                    )

    results = average_pq(pq_stat, categories)
    with open(resultsFile, 'w') as f:
        logger.info("Saving computed results in {}".format(resultsFile))
        json.dump(results, f, sort_keys=True, indent=4)
    print_results(results, categories, logger)

    t_delta = time.time() - start_time
    logger.info("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


# The main method
def main():
    cityscapesPath = os.environ.get(
        'CITYSCAPES_DATASET', os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    )
    gtJsonFile = os.path.join(cityscapesPath, "gtFine", "cityscapes_panoptic_val.json")

    predictionPath = os.environ.get(
        'CITYSCAPES_RESULTS',
        os.path.join(cityscapesPath, "results")
    )
    predictionJsonFile = os.path.join(predictionPath, "cityscapes_panoptic_val.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json-file",
                        dest="gtJsonFile",
                        help= '''path to json file that contains ground truth in COCO panoptic format.
                            By default it is $CITYSCAPES_DATASET/gtFine/cityscapes_panoptic_val.json.
                        ''',
                        default=gtJsonFile,
                        type=str)
    parser.add_argument("--gt-folder",
                        dest="gtFolder",
                        help= '''path to folder that contains ground truth *.png files. If the
                            argument is not provided this script will look for the *.png files in
                            'name' if --gt-json-file set to 'name.json'.
                        ''',
                        default=None,
                        type=str)
    parser.add_argument("--prediction-json-file",
                        dest="predictionJsonFile",
                        help='''path to json file that contains prediction in COCO panoptic format.
                            By default is either $CITYSCAPES_RESULTS/cityscapes_panoptic_val.json
                            or $CITYSCAPES_DATASET/results/cityscapes_panoptic_val.json if
                            $CITYSCAPES_RESULTS is not set.
                        ''',
                        default=predictionJsonFile,
                        type=str)
    parser.add_argument("--prediction-folder",
                        dest="predictionFolder",
                        help='''path to folder that contains prediction *.png files. If the
                            argument is not provided this script will look for the *.png files in
                            'name' if --prediction-json-file set to 'name.json'.
                        ''',
                        default=None,
                        type=str)
    resultFile = "resultPanopticSemanticLabeling.json"
    parser.add_argument("--results_file",
                        dest="resultsFile",
                        help="File to store computed panoptic quality. Default: {}".format(resultFile),
                        default=resultFile,
                        type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.gtJsonFile):
        printError("Could not find a ground truth json file in {}. Please run the script with '--help'".format(args.gtJsonFile))
    if args.gtFolder is None:
        args.gtFolder = os.path.splitext(args.gtJsonFile)[0]

    if not os.path.isfile(args.predictionJsonFile):
        printError("Could not find a prediction json file in {}. Please run the script with '--help'".format(args.predictionJsonFile))
    if args.predictionFolder is None:
        args.predictionFolder = os.path.splitext(args.predictionJsonFile)[0]

    evaluatePanoptic(args.gtJsonFile, args.gtFolder, args.predictionJsonFile, args.predictionFolder, args.resultsFile)

    return

# call the main method
if __name__ == "__main__":
    main()
