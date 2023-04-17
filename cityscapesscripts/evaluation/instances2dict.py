#!/usr/bin/python
#
# Convert instances from png files to a dictionary
#

from __future__ import print_function, absolute_import, division
import os, sys

# Cityscapes imports
from cityscapesscripts.evaluation.instance import *
from cityscapesscripts.helpers.csHelpers import *
from mmseg.datasets.utils import resize_with_pad
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def instances2dict(imageFileList, verbose=False, dataset_name=None, rgb2id=None, input_image_size=None, mapillary_dataloading_style='OURS', debug=False):
    imgCount     = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName) # loading the GT instance ground turth e.g. data/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_instanceIds.png

        if 'mapillary' in dataset_name:
            if mapillary_dataloading_style == 'DADA':
                raise NotImplementedError('To evaluate the mapillary on original image shape for panoptic seg,'
                                          ' you need to first upsample the predicted masks with pad_with_fixed_AS(). '
                                          'This part is not implemented yet.')
                # target_ratio = 1024 / 768
                # img = pad_with_fixed_AS(target_ratio, img, fill_value=0, is_label=False)
            else:
                img, new_image_shape = resize_with_pad(img, [1024, 768], Image.NEAREST, pad_value=0, is_label=True) # resize_with_pad() returns numpy array,
                                                                                                                # so no need to convert
            imgNp = rgb2id(img).astype(np.uint32)   # since we dont keep separate gt instance ids like cityscapes (e.g. *_gtFine_instanceIds.png) for mapillary
                                                    # we generate the gt instance id maps from the panoptic color PNG images

        elif 'cityscapes' in dataset_name:
            # Image as numpy array
            if debug:
                img = img.resize((1024, 512), Image.NEAREST)
            imgNp = np.array(img)
        else:
            NotImplementedError('no implementation found at def instances2dict(...) --> cityscapesscripts/evaluation/instances2dict.py')

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            instanceObj = Instance(imgNp, instanceId)
            instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

def main(argv):
    fileList = []
    if (len(argv) > 2):
        for arg in argv:
            if ("png" in arg):
                fileList.append(arg)
    instances2dict(fileList, True)

if __name__ == "__main__":
    main(sys.argv[1:])
