# Obtained from: https://github.com/vikolss/DACS

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks_v3(labels, multi_thing_classes=False):
    class_masks = []
    cityscapes_thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
    thing_classes_in_mixed_img = True
    check_falg = False
    for label in labels:
        classes = torch.unique(labels)
        classes_things = torch.tensor([c for c in classes if c in cityscapes_thing_list])
        classes_things = classes_things.to(classes.device)
        classes_things = classes_things.to()
        nclasses_things = classes_things.shape[0]
        if nclasses_things > 0:
            if multi_thing_classes:
                num_thing_cls = int((nclasses_things + nclasses_things % 2) / 2)
            else:
                num_thing_cls = 1
            class_choice_thing = np.random.choice(nclasses_things, num_thing_cls, replace=False)
            classes_things = classes_things[torch.Tensor(class_choice_thing).long()]
            class_masks.append(generate_class_mask(label, classes_things).unsqueeze(0))
        elif nclasses_things == 0:
            if not check_falg:
                thing_classes_in_mixed_img = False
                check_falg = True
            nclasses = classes.shape[0]
            class_choice = np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
            classes = classes[torch.Tensor(class_choice).long()]
            class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
        else:
            raise NotImplementedError('Error in function get_class_masks_v3() in mmseg/models/utils/dacs_transforms.py !!!')
    return class_masks, thing_classes_in_mixed_img


def get_class_masks_v2(labels):
    class_masks = []
    cityscapes_thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
    thing_classes_in_mixed_img = True
    check_falg = False
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
        # check if thing class is present
        thing_classes_sampled = torch.tensor([c for c in classes if c in cityscapes_thing_list])
        num_thing_classes = thing_classes_sampled.shape[0]
        if num_thing_classes == 0 and not check_falg:
            check_falg = True
            thing_classes_in_mixed_img = False
    return class_masks, thing_classes_in_mixed_img


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label, classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
