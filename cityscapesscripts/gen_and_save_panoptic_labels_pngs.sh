#!/bin/bash

cd /path/to/the/edaps
source ~/venv/edaps/bin/activate
PYTHONPATH="</path/to/the/edaps>:$PYTHONPATH" && export PYTHONPATH
python cityscapesscripts/preparation/createPanopticImgs.py