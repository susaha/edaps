#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode ground truth classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 
# With this tool, you can generate option
#   d) *labelTrainIds.png     : the class is encoded by its training ID
# This encoding might come handy for training purposes. You can use
# the file labels.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Uses the converter tool in 'json2labelImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys

# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.preparation.json2labelImg import json2labelImg


os.environ['CITYSCAPES_DATASET'] = '/media/suman/CVLHDD/apps/datasets/cityscapes_4_panoptic_deeplab/cityscapes'

# The main method
def main():
    # Where to look for Cityscapes
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    # how to search for all ground truth
    # searchFine   = os.path.join( cityscapesPath , "gtFine"   , "train" , "*" , "*_gt*_polygons.json" ) # use this if you want to extract only the train files
    searchFine = os.path.join(cityscapesPath, "gtFine", "*", "*", "*_gt*_polygons.json")  # use this for extracting all (train, val and test) files
    # searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" ) # original code

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    # filesCoarse = glob.glob( searchCoarse )       # original code
    # filesCoarse.sort()                            # original code

    # concatenate fine and coarse
    files = filesFine
    # files = filesFine + filesCoarse               # original code
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        printError( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace( "_polygons.json" , "_labelTrainIds_synthia_to_cityscapes_16_cls.png" )  # modifed for generating synthia_to_cityscapes_16_cls compatible PNG label images
        # dst = f.replace("_polygons.json", "_labelTrainIds.png")   # original code

        # do the conversion
        try:
            json2labelImg( f , dst , "trainIds" )
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()
