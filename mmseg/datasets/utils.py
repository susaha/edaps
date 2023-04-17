# taken from https://github.com/valeoai/DADA
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def pad_with_fixed_AS(target_ratio, image, fill_value=0, is_label=False):
    dimW = float(image.size[0])
    dimH = float(image.size[1])

    image_ratio = dimW / dimH

    if target_ratio > image_ratio:
        dimW = target_ratio * dimH
    elif target_ratio < image_ratio:
        dimH = dimW / target_ratio
    else:
        if is_label:
            return np.array(image, dtype=np.uint8)
        else:
            return np.array(image)

    if is_label:
        image = np.array(image, dtype=np.uint8)
    else:
        image = np.array(image)

    result = np.ones((int(dimH), int(dimW), int(image.shape[2])), np.float32) * fill_value

    placeholder = result[:image.shape[0], :image.shape[1]]

    placeholder[:] = image

    return result


def resize_with_pad(image, target_size, resize_type, pad_value=0, is_label=False):
    '''
    image must be a PIL image
    target_size: a list or tuple, e.g. [1024, 768]
    resize_type: Image.BICUBIC for image, Image.NEAREST for GT panoptic or semantic labels
    pad_value: 0 , useed to fill the padded region
    is_label: set it True for GT  labels

    for images, call:       resize_with_pad(results, Image.BICUBIC, pad_value=self.img_pad_value)
    for gt labels, call:    resize_with_pad(results, Image.NEAREST, pad_value=self.label_pad_value, is_label=True)

    Note: when is_label=True I am using np.uint32, this datatype is suitable for
    semanitc, panoptic and instance GT labels.
    For semanitc, panoptic GT labels uint8 is fine because they have values between 0 and 255.
    Since I am using this function within the cityscapescript evalscript which loads the GT instance map
    which stores the panoptic ids larger than 255, I can not use uint8, so used uint32 when is_label=True
    semanitc GT labels have values 0,1,2,..,18, 2555
    panoptic GT labes are converted color PNG images, we use rgb2id to convert the color code to panoptic id
    In rgb2id() the npuint8 is converted to np.uint32 to stroe the values > 255
    If you have any GT labels for example GT instance labels which have the actulpanoptic ids
    then you need to use uint32 instead uint8

    '''

    # find which size to fit to the target size
    target_ratio = target_size[0] / target_size[1]
    image_ratio = image.size[0] / image.size[1]
    if image_ratio > target_ratio:
        resize_ratio = target_size[0] / image.size[0]  # target_widht / image_widht
        new_image_shape = (target_size[0], int(image.size[1] * resize_ratio))
    else:
        resize_ratio = target_size[1] / image.size[1]  # target_height / image_height
        new_image_shape = (int(image.size[0] * resize_ratio), target_size[1])
    image_resized = image.resize(new_image_shape, resize_type)
    if is_label:
        image_resized = np.array(image_resized, dtype=np.uint32)
    else:
        image_resized = np.array(image_resized)
    if image_resized.ndim == 2:
        image_resized = image_resized[:, :, None]
    result = np.ones(target_size[::-1] + [image_resized.shape[2], ], np.float32) * pad_value
    assert image_resized.shape[0] <= result.shape[0]
    assert image_resized.shape[1] <= result.shape[1]
    placeholder = result[:image_resized.shape[0], :image_resized.shape[1]]
    placeholder[:] = image_resized

    return result, new_image_shape



