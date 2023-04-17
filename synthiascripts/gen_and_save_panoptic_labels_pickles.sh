#!/bin/bash

cd /path/to/the/edaps
source ~/venv/edaps/bin/activate
PYTHONPATH="</path/to/the/edaps>:$PYTHONPATH" && export PYTHONPATH
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 0 --max 1000 &
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 1000 --max 2000  &
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 2000 --max 3000  &
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 3000 --max 4000  &
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 4000 --max 5000  &
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 5000 --max 6000  &
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 6000 --max 7000  &
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 7000 --max 8000  &
python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 8000 --max 10000