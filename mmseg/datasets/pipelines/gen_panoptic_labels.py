# --------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted from: https://github.com/bowenc0221/panoptic-deeplab
# --------------------------------------------------------------------------------

import numpy as np
from ..builder import PIPELINES
from cityscapesscripts.helpers.labels import id2label
from tools.panoptic_deeplab.utils import rgb2id


@PIPELINES.register_module()
class GenPanopLabels(object):

    def __init__(self, sigma, mode):
        self.ignore_label = 255
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.ignore_stuff_in_offset = True
        self.small_instance_area = 4096 # not using currently
        self.small_instance_weight = 3
        self.ignore_crowd_in_semantic = True
        self.sigma = sigma
        self.mode = mode
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, results):
        panoptic = results['gt_panoptic_seg']
        segments = results['ann_info']['segments_info']
        panoptic = rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        instance = np.zeros_like(panoptic, dtype=np.uint8)
        foreground = np.zeros_like(panoptic, dtype=np.uint8)
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(panoptic, dtype=np.float32)
        x_coord = np.ones_like(panoptic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        instance_id = 1
        for seg in segments:
            cat_id = seg["category_id"]
            if self.mode == 'val':
                labelInfo = id2label[cat_id]
                cat_id = labelInfo.trainId
            if self.ignore_crowd_in_semantic:
                if not seg['iscrowd']:
                    semantic[panoptic == seg["id"]] = cat_id
            else:
                semantic[panoptic == seg["id"]] = cat_id
            if cat_id in self.thing_list:
                instance[panoptic == seg["id"]] = instance_id
                instance_id += 1
            if not seg['iscrowd']:
                center_weights[panoptic == seg["id"]] = 1
                if self.ignore_stuff_in_offset:
                    if cat_id in self.thing_list:
                        offset_weights[panoptic == seg["id"]] = 1
                else:
                    offset_weights[panoptic == seg["id"]] = 1
            if cat_id in self.thing_list:
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    continue
                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])
                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(
                    center[0, aa:bb, cc:dd], self.g[a:b, c:d])
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]
        results['gt_semantic_seg'] = semantic.astype('long')
        results['seg_fields'].append('gt_semantic_seg')
        results['gt_center'] = center.astype(np.float32)
        results['seg_fields'].append('gt_center')
        results['gt_offset'] = offset.astype(np.float32)
        results['seg_fields'].append('gt_offset')
        results['center_weights'] = center_weights.astype(np.float32)
        results['seg_fields'].append('center_weights')
        results['offset_weights'] = offset_weights.astype(np.float32)
        results['seg_fields'].append('offset_weights')
        results['gt_instance_seg'] = instance.astype('long')
        results['seg_fields'].append('gt_instance_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(ignore_label={self.ignore_label}, ' \
                    f'(thing_list={self.thing_list}, ' \
                    f'(ignore_stuff_in_offset={self.ignore_stuff_in_offset}, ' \
                    f'(small_instance_area={self.small_instance_area}, ' \
                    f'(small_instance_weight={self.small_instance_weight}, ' \
                    f'(sigma={self.sigma}, ' \
                    f'(g={self.g}, '
        return repr_str

