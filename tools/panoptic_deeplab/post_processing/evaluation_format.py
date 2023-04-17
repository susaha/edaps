# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted from: https://github.com/bowenc0221/panoptic-deeplab
# Modifications: Support for panoptic segmentation evaluation
# ------------------------------------------------------------------------------------


from collections import OrderedDict
import numpy as np
import torch
import statistics as st


def py_nms(dets, thresh=0.5):
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def get_cityscapes_instance_format_for_maskrcnn_v3(
        boxes, masks, pred_shape=(1024, 2048), mask_score_th=0.0,
        sem_seg=None, device=None, thing_list=None,
        use_semantic_decoder_for_instance_labeling=False,
        use_semantic_decoder_for_panoptic_labeling=False,
        nms_th=0.5,
        intersec_th=0.3,
        ):

    label_divisor, ignore_label = 1000, 255
    void_label = label_divisor * ignore_label
    thing_list_mapids = {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18}
    assert len(boxes) == len(masks), 'boxes and masks lists must have same length!'
    num_classes = len(boxes)
    instances = []
    ins_seg = np.zeros(pred_shape).astype(int)
    pan_seg_thing_classes = np.zeros(pred_shape).astype(int) + void_label
    ins_seg_all = np.zeros(pred_shape).astype(int)
    ins_cnt = 1
    ins_cnt_visual = 1
    bboxes_visuals = []
    dict_v1 = {}
    bid = 0
    for c in range(num_classes):
        boxes_c = boxes[c]
        if boxes_c.shape[0] > 0:
            masks_c = masks[c]
            assert boxes_c.shape[0] == len(masks_c), 'there must be same number of masks and boxes for a class!'
            for m in range(len(masks_c)):
                box  = boxes_c[m]
                mask  = masks_c[m]
                cid = thing_list_mapids[c]
                dict_v1[bid] = (cid, box, mask)
                bid += 1
    all_boxes = np.zeros((len(dict_v1), 5)).astype(float)
    for bid in dict_v1.keys():
        all_boxes[bid, :] = dict_v1[bid][1]
    keep = py_nms(all_boxes, thresh=nms_th)
    dict_v2 = {}
    for bid in keep:
        for cid in thing_list:
            if cid == dict_v1[bid][0]:
                box_score = dict_v1[bid][1][4]
                if  box_score > mask_score_th[str(cid)]:
                    if cid not in dict_v2:
                        dict_v2[cid] = []
                        dict_v2[cid].append((box_score, dict_v1[bid][2]))
                    else:
                        current_mask = dict_v1[bid][2]
                        for (prev_box_score, prev_mask) in dict_v2[cid]:
                            assert prev_box_score >= box_score
                            intersection = np.count_nonzero(np.logical_and(current_mask, prev_mask))
                            if intersection <= intersec_th:
                                mask_temp = np.zeros(pred_shape).astype(int)
                                mask_temp[np.logical_and(current_mask, ~prev_mask)] = 1
                                mask_temp = mask_temp.astype(bool)
                                dict_v2[cid].append((box_score, mask_temp))

    class_id_tracker = {}
    for cid in dict_v2.keys():
        for (score, mask) in dict_v2[cid]:
            ins = OrderedDict()
            ins['pred_class'] = cid
            ins['pred_mask'] = np.array(mask, dtype='uint8')
            ins['score'] = score
            instances.append(ins)
            ins_seg[mask] = ins_cnt
            if cid in class_id_tracker:
                new_ins_id = class_id_tracker[cid]
            else:
                class_id_tracker[cid] = 1
                new_ins_id = 1
            class_id_tracker[cid] += 1
            pan_seg_thing_classes[mask] = cid * label_divisor + new_ins_id
            ins_cnt += 1
    return instances, ins_seg, pan_seg_thing_classes


def get_cityscapes_instance_format_for_maskrcnn(
                                                    boxes,
                                                    masks,
                                                    pred_shape=(1024, 2048),
                                                    mask_score_th=0.0,
                                                    sem_seg=None,
                                                    device=None,
                                                    thing_list=None,
                                                    use_semantic_decoder_for_instance_labeling=False,
                                                    use_semantic_decoder_for_panoptic_labeling=False,
                                                    nms_th=0.5,
                                                    intersec_th=0.3,
                                                    ):

    thing_list_mapids = {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18}
    assert len(boxes) == len(masks), 'boxes and masks lists must have same length!'
    num_classes = len(boxes)
    instances = []
    ins_seg = np.zeros(pred_shape).astype(int)
    ins_cnt = 1
    for c in range(num_classes):
        boxes_c = boxes[c]
        if boxes_c.shape[0] > 0:
            masks_c = masks[c]
            assert boxes_c.shape[0] == len(masks_c), 'there must be same number of masks and boxes for a class!'
            for m in range(len(masks_c)):
                mask_score = boxes_c[m, 4]
                ins = OrderedDict()
                ins['pred_class'] = thing_list_mapids[c]
                ins['pred_mask'] = np.array(masks_c[m], dtype='uint8')
                ins['score'] = mask_score
                instances.append(ins)
                if mask_score >= mask_score_th:
                    ins_seg[masks_c[m]] = ins_cnt
                    ins_cnt += 1
    return instances, ins_seg, None

def get_cityscapes_instance_format_for_maskformer(boxes, masks):
    assert len(boxes) == len(masks), 'boxes and masks lists must have same length!'
    num_classes = len(boxes)
    instances = []
    for c in range(num_classes):
        if c >= 11: # thing classes are from 11 to 18
            boxes_c = boxes[c]
            if boxes_c.shape[0] > 0:
                masks_c = masks[c]
                assert boxes_c.shape[0] == len(masks_c), 'there must be same number of masks and boxes for a class!'
                for m in range(len(masks_c)): # loop over masks of class c
                    ins = OrderedDict()
                    ins['pred_class'] = c
                    ins['pred_mask'] = np.array(masks_c[m], dtype='uint8')
                    ins['score'] = boxes_c[m, 4]
                    instances.append(ins)
    return instances


def get_cityscapes_instance_format(panoptic, sem, ctr_hmp, label_divisor, score_type="semantic"):
    """
    Get Cityscapes instance segmentation format.
    Arguments:
        panoptic: A Numpy Ndarray of shape [H, W].
        sem: A Numpy Ndarray of shape [C, H, W] of raw semantic output.
        ctr_hmp: A Numpy Ndarray of shape [H, W] of raw center heatmap output.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        score_type: A string, how to calculates confidence scores for instance segmentation.
            - "semantic": average of semantic segmentation confidence within the instance mask.
            - "instance": confidence of heatmap at center point of the instance mask.
            - "both": multiply "semantic" and "instance".
    Returns:
        A List contains instance segmentation in Cityscapes format.
    """
    instances = []

    pan_labels = np.unique(panoptic)
    for pan_lab in pan_labels:
        if pan_lab % label_divisor == 0:
            # This is either stuff or ignored region.
            continue

        ins = OrderedDict()

        train_class_id = pan_lab // label_divisor
        ins['pred_class'] = train_class_id

        mask = panoptic == pan_lab
        ins['pred_mask'] = np.array(mask, dtype='uint8')

        sem_scores = sem[train_class_id, ...]
        ins_score = np.mean(sem_scores[mask])
        # mask center point
        mask_index = np.where(panoptic == pan_lab)
        center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
        ctr_score = ctr_hmp[int(center_y), int(center_x)]

        if score_type == "semantic":
            ins['score'] = ins_score
        elif score_type == "instance":
            ins['score'] = ctr_score
        elif score_type == "both":
            ins['score'] = ins_score * ctr_score
        else:
            raise ValueError("Unknown confidence score type: {}".format(score_type))

        instances.append(ins)

    return instances
