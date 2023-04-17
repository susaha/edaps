#!/bin/bash

cd /path/to/the/edaps
source ~/venv/edaps/bin/activate
PYTHONPATH="</path/to/the/edaps>:$PYTHONPATH" && export PYTHONPATH
python mapillaryscripts/save_panoptic_gt_labels_for_mapillary_as_pickle_files_19cls.py