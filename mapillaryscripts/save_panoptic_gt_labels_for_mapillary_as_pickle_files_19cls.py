# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

'''
NOTES:
    When we run mapillary demo script, the class ids are starting from 0 to 65.
    But in the json file (panoptic/panoptic_2018.json), the category ids are from 1 to 66.
    We need to modify your script according to this.

    You can see this in the CVRN code:
        mapillaryscripts/random_exp/init_vistas2cityscapes_format.py
        line no. 71 and 72 as given below:
            for k, v in id_to_trainid.items():
                label_copy[label == k+1] = v
        note, it uses k+1

    Also note, the ground truth label PNGs for semantic segmention,
    they encode class ids from 0 to 65.
'''

from __future__ import print_function
import json
import os.path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing
import functools
import traceback
# from mapillaryscripts.map_ids import get_map_m2c
from mapillaryscripts.map_ids_19_classes import get_map_m2c
import imageio
import pickle
from datetime import datetime
from statistics import mean


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

# single core processing
@get_traceback
def compute_single_core(proc_id, img_range, annotations,  base_path, split, map_m2c, VIS):
    cityscapes_thing_list = [24, 25, 26, 27, 28, 31, 32, 33]
    panop_path = os.path.join(base_path, split, 'panoptic')
    data_out = 'data/mapillary'
    out_path = os.path.join(data_out, split, 'panoptic_mapped_to_cityscapes_ids_19_classes', 'pickles')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i, anno in enumerate(annotations):
        # check if the image belongs to this process
        if i < img_range[0] or i > img_range[1]:
            continue
        seg_info = anno['segments_info']
        image_id = anno['image_id']
        pan_fname = os.path.join(panop_path, '{}.png'.format(image_id))
        pan = Image.open(pan_fname)
        # convert PIL to numpy array
        pan = np.array(pan).astype(np.uint32)
        # convert the RGB color code (R,G,B) to unique panoptic ids - this is same as the cityscpes rgb2id() function
        # this converts the color PNG of shape HxWx3 (i.e., pan) --> to panoptic id matrix of shape HxW (i.e., pan_id_array)
        pan_id_array = pan[:, :, 0] + (2 ** 8) * pan[:, :, 1] + (2 ** 16) * pan[:, :, 2]
        # init array to store the new panoptic label ids in cityscapes style
        if not VIS:
            panoptic_cityscapes_style = np.zeros((pan_id_array.shape[0], pan_id_array.shape[1]), dtype=np.uint32)
            panoptic_cityscapes_style_outfile = os.path.join(out_path, '{}.pkl'.format(image_id))
        else:
            panoptic_cityscapes_style = np.zeros((pan_id_array.shape[0], pan_id_array.shape[1]), dtype=np.uint8)
            panoptic_cityscapes_style_outfile = os.path.join(out_path, '{}.png'.format(image_id))
        # initialize a dictionary to store class speific segment count for thing classes for each image
        segment_count = {}
        for id in cityscapes_thing_list:
            segment_count[id] = 0
        # loop over unique segments in an image
        for seg in seg_info:
            segment_in_19cls_set = True
            # convert the mapillary class ids to cityscapes class ids
            try:
                cat_id = map_m2c[seg['category_id'] - 1] #  to know why we deduct 1 here --> refer to the above NOTE
            except KeyError:
                segment_in_19cls_set = False
            isCrowd = seg['iscrowd']
            panoptic_id = seg['id']
            mask = pan_id_array == panoptic_id # compute the segment boolean mask
            assert seg['area'] == np.sum(mask), 'area of the segment in seg is different than our computed segment area!'
            # compute the cityscapes style panoptic id
            if segment_in_19cls_set:
                if isCrowd == 0:
                    if cat_id in cityscapes_thing_list:
                        new_panoptic_id = cat_id * 1000 + segment_count[cat_id]
                        segment_count[cat_id] += 1
                    else:
                        new_panoptic_id = cat_id
                # if segment in crowd region then both stuff and thing class ids are < 1000
                else:
                    new_panoptic_id = cat_id
                panoptic_cityscapes_style[mask] = new_panoptic_id
        if VIS:
            imageio.imwrite(panoptic_cityscapes_style_outfile, panoptic_cityscapes_style, format='PNG-FI')
        else:
            with open(panoptic_cityscapes_style_outfile, 'wb') as f:
                pickle.dump(panoptic_cityscapes_style, f)
        if i % 100 == 0:
            print('proc_id: {}, processed imgid: {}, img rang [start: {}, end: {}]'.format(proc_id, i, img_range[0], img_range[1]))

def main():
    DEBUG = False
    VIS = False
    NUM_CPUS_TO_FREE = 4
    # swtich to train/val
    split = 'train' # TODO
    # split = 'val' # TODO
    # get the map_mapillary_to_cityscapes dictionary
    map_m2c = get_map_m2c()
    base_path = 'datasets/Mapillary-Vistas-v1.2'
    json_fname = os.path.join(base_path, split, 'panoptic/panoptic_2018.json')
    # read in panoptic file
    with open(json_fname) as panoptic_file:
        panoptic = json.load(panoptic_file)
    if DEBUG:
        compute_single_core(0, [0, 100], panoptic['annotations'], base_path, split, map_m2c, VIS)
    else:
        # multi core processining
        num_imgs = len(panoptic['annotations'])
        # keeping two cpu free on my local desktop
        num_cpus = multiprocessing.cpu_count() - NUM_CPUS_TO_FREE
        print("Number of cores: {}".format(num_cpus))
        # num imgs per cpu
        num_imgs_per_cpu = int(num_imgs / num_cpus)
        img_range_list = []
        start = 0
        end = num_imgs_per_cpu - 1
        for i in range(num_cpus):
            img_range_list.append([start, end])
            start = end + 1
            end += num_imgs_per_cpu
        # set a big no. so that for the last batch, it considers all the images at the end
        img_range_list[-1][1] = 100000
        workers = multiprocessing.Pool(processes=num_cpus)
        for proc_id, img_range in enumerate(img_range_list):
            workers.apply_async(compute_single_core, (proc_id, img_range, panoptic['annotations'], base_path, split, map_m2c, VIS))
        workers.close()
        workers.join()

if __name__ == "__main__":
    main()