# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import numpy as np
import os
from datetime import datetime
import torch


def rgb2id(color):
    """Converts the color to panoptic label.
    Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
    Args:
        color: Ndarray or a tuple, color encoded image.
    Returns:
        Panoptic label.
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])



def create_label_colormap_16cls():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]  # road; 804080ff
    colormap[1] = [244, 35, 232]  # sidewalk; f423e8ff
    colormap[2] = [70, 70, 70]  # building; 464646ff
    colormap[3] = [102, 102, 156]  # wall; 666699ff
    colormap[4] = [190, 153, 153]  # fence; be9999ff
    colormap[5] = [153, 153, 153]  # pole; 999999ff
    colormap[6] = [250, 170, 30]  # traffic-light; faaa1eff
    colormap[7] = [220, 220, 0]  # traffic-sign; dcdc00ff
    colormap[8] = [107, 142, 35]  # vegetation; 6b8e23ff
    colormap[9] = [70, 130, 180]  # sky; 4682b4ff
    colormap[10] = [220, 20, 60]  # person; dc143cff
    colormap[11] = [255, 0, 0]  # rider; ff0000ff
    colormap[12] = [0, 0, 142]  # car; 00008eff
    colormap[13] = [0, 60, 100]  # bus; 003c64ff
    colormap[14] = [0, 0, 230]  # motocycle, 0000e6ff
    colormap[15] = [119, 11, 32]  # bicycle, 770b20ff
    return colormap

def create_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def create_panop_eval_folders(panoptic_eval_root_folder, panop_eval_temp_folder=None):
    '''
    We have implemented a feature in our Git repository that generates a new folder with a unique name for each panoptic evaluation.
    This ensures that each evaluation can be performed independently without interference from any previous evaluations.
    When the evaluation is complete, the folder and its contents can be safely deleted, and a new root folder and subfolders can be generated for the next evaluation.
    It is important to note that creating and deleting a large root folder may take a few seconds to complete due to the size of the generated PNG and text files.
    Therefore, to avoid any potential overlap or delays caused by the deletion process, we generate a new root folder with a unique name for each evaluation.
    This ensures that a new, independent evaluation can be initiated immediately without any waiting period.
    '''

    os.makedirs(panoptic_eval_root_folder, exist_ok=True)
    if not panop_eval_temp_folder:
        # str1 = datetime.now().strftime("%m-%Y")
        str2 = datetime.now().strftime("%d-%m-%Y")
        str3 = datetime.now().strftime("%H-%M-%S-%f")
        panop_eval_temp_folder = 'panop_eval_{}_{}'.format(str2, str3)
    panop_eval_temp_folder_abs_path = os.path.join(panoptic_eval_root_folder, panop_eval_temp_folder)
    panop_eval_folder_dict = {}
    panop_eval_folder_dict['semantic'] = os.path.join(panop_eval_temp_folder_abs_path, 'semantic')
    panop_eval_folder_dict['instance'] = os.path.join(panop_eval_temp_folder_abs_path, 'instance')
    panop_eval_folder_dict['panoptic'] = os.path.join(panop_eval_temp_folder_abs_path, 'panoptic')
    panop_eval_folder_dict['visuals'] = os.path.join(panop_eval_temp_folder_abs_path, 'visuals')
    os.makedirs(panop_eval_folder_dict['semantic'],  exist_ok=True)
    os.makedirs(panop_eval_folder_dict['instance'],  exist_ok=True)
    os.makedirs(panop_eval_folder_dict['panoptic'],  exist_ok=True)
    os.makedirs(panop_eval_folder_dict['visuals'], exist_ok=True)
    return panop_eval_temp_folder_abs_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_loss_info_str(loss_meter_dict):
    msg = ''
    for key in loss_meter_dict.keys():
        msg += '{name}: {meter.val:.6f} ({meter.avg:.6f})\t'.format(name=key, meter=loss_meter_dict[key])

    return msg


def to_cuda(batch, device):
    if type(batch) == torch.Tensor:
        batch = batch.to(device)
    elif type(batch) == dict:
        for key in batch.keys():
            batch[key] = to_cuda(batch[key], device)
    elif type(batch) == list:
        for i in range(len(batch)):
            batch[i] = to_cuda(batch[i], device)
    return batch


def get_module(model, distributed):
    if distributed:
        return model.module
    else:
        return model