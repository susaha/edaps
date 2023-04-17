# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np
from PIL import Image
import imageio
import pickle
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import id2label, labels
from datetime import datetime


# The main method
def convert2panoptic(pickle_path=None, outputFolder=None, useTrainId=None, BBOX_VIS=None):
    categories = []
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                           'name': label.name,
                           'color': label.color,
                           'supercategory': label.category,
                           'isthing': 1 if label.hasInstances else 0})
    file_list = os.listdir(pickle_path)
    file_list.sort()
    # a bit verbose
    print("Converting {} annotation files".format(len(file_list)))
    outputBaseFile = "panoptic"
    outFile = os.path.join(outputFolder, "{}.json".format(outputBaseFile))
    print("Json file with the annotations in panoptic format will be saved in {}".format(outFile))
    panopticFolder = os.path.join(outputFolder, outputBaseFile)
    if not os.path.isdir(panopticFolder):
        print("Creating folder {} for panoptic segmentation PNGs".format(panopticFolder))
        os.mkdir(panopticFolder)
    print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))
    images = []
    annotations = []
    for progress, f in enumerate(file_list):
        if BBOX_VIS:
            lineWidth = 1
            cc = 'r'
            fig, ax = plt.subplots(1, 2)
            bbox_vis = []
        f = os.path.join(pickle_path, f)
        with open(f,'rb') as f2o:
            originalFormat = pickle.load(f2o)
        fileName = os.path.basename(f)
        imageId = fileName.replace(".pkl", "")
        inputFileName = fileName.replace(".pkl", ".png")
        outputFileName = inputFileName
        images.append({"id": imageId, "width": int(originalFormat.shape[1]), "height": int(originalFormat.shape[0]), "file_name": inputFileName})
        pan_format = np.zeros((originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8)
        segmentIds = np.unique(originalFormat)
        segmInfo = []
        for segmentId in segmentIds:
            if segmentId < 1000:
                semanticId = segmentId
                isCrowd = 1
            else:
                semanticId = segmentId // 1000
                isCrowd = 0
            labelInfo = id2label[semanticId]
            categoryId = labelInfo.trainId if useTrainId else labelInfo.id
            if labelInfo.ignoreInEval:
                continue
            if not labelInfo.hasInstances:
                isCrowd = 0
            mask = originalFormat == segmentId
            color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
            pan_format[mask] = color
            area = np.sum(mask) # segment area computation
            # bbox computation for a segment
            hor = np.sum(mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]
            if BBOX_VIS:
                bbox_vis.append(bbox)
            segmInfo.append({"id": int(segmentId),
                             "category_id": int(categoryId),
                             "area": int(area),
                             "bbox": bbox,
                             "iscrowd": isCrowd})
        annotations.append({'image_id': imageId,
                            'file_name': outputFileName,
                            "segments_info": segmInfo})
        Image.fromarray(pan_format).save(os.path.join(panopticFolder, outputFileName))
        if BBOX_VIS:
            img = Image.fromarray(pan_format)
            ax[0].imshow(img)
            ax[1].imshow(img)
            for bb in bbox_vis:
                x = bb[0]
                y = bb[1]
                w = bb[2]
                h = bb[3]
                rect = patches.Rectangle((x, y), w, h, linewidth=lineWidth, edgecolor=cc, facecolor='none')
                ax[1].add_patch(rect)
            file_name = os.path.join(outputFolder, '{}_bbox.png'.format(imageId))
            plt.savefig(file_name, dpi=300)
            plt.close()
        print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(file_list)), end=' ')
        sys.stdout.flush()
    print("\nSaving the json file {}".format(outFile))
    d = {'images': images,
         'annotations': annotations,
         'categories': categories}
    with open(outFile, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="pickle_path",
                        help="path to the Cityscapes dataset 'gtFine' folder",
                        default=None,
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default=None,
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["val", "train", "test"],
                        type=str)
    args = parser.parse_args()
    data_out = 'data/mapillary'
    # TODO: USER INPUTS BELOW
    DEBUG = False
    # split='train'
    split = 'val'
    if split == 'train':
        useTrainId = True
    else:
        useTrainId = False
    # if you want to visualize the boxes around each segments for debug purpose
    BBOX_VIS = False
    if DEBUG:
        args.pickle_path = ''
        args.outputFolder = ''
    else:
        args.pickle_path = f'{data_out}/{split}/panoptic_mapped_to_cityscapes_ids_19_classes/pickles'
        args.outputFolder = f'{data_out}/{split}/panoptic_mapped_to_cityscapes_ids_19_classes/pngs'
    if not os.path.exists(args.outputFolder):
        os.makedirs(args.outputFolder)
    convert2panoptic(args.pickle_path, args.outputFolder, useTrainId=useTrainId, BBOX_VIS=BBOX_VIS)


if __name__ == "__main__":
    main()