#!/bin/bash

cd /path/to/the/edaps
source ~/venv/edaps/bin/activate
PYTHONPATH="</path/to/the/edaps>:$PYTHONPATH" && export PYTHONPATH
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_color_png_file_19cls.py