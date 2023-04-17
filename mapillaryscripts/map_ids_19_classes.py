# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

'''
NOTES:
    # _CITYSCAPES_THING_LIST = [24, 25, 26, 27, 28, 31, 32, 33] # note these are ids and not the trainIds

    Mapillary ids are from mapillaryscripts/mapillary-class-names-and-ids-v1.2.txt.
    These ids are 0 indexed i.e. from 0 to 65
    The ids in the panoptic jsonfile ('panoptic/panoptic_2018.json') are 1 indexed i.e. from 1 to 66.
    whenever you read the segnent info from the JSON file, deduct 1 from the categoryid.

    CVRN Mapillary to Cityscapes 19 class mapping.
        I copied the following lines of code as a reference from:
        https://github.com/jxhuang0508/CVRN/blob/d49a2565be45f9c750dc59c44afce57fb768b5e1/data%20loader%20and%20processing/init_vistas2cityscapes_format.py#L1
        script file path = CVRN/data loader and processing/init_vistas2cityscapes_format.py
            id_to_trainid = {7: 0, 8: 0, 10: 0, 13: 0, 14: 0, 23: 0, 24: 0,
                                2: 1, 9: 1, 11: 1, 15: 1, 17: 2, 6: 3, 3: 4,
                                45: 5, 47: 5, 48: 6, 49: 7, 50: 7, 30: 8, 29: 9,
                                27: 10, 19:11, 20:12, 21:12, 22:12, 55:13, 61:14, 54:15, 58:16, 57:17, 52:18,
                                0: 255, 1: 255, 4: 255, 5: 255, 12: 255, 16: 255, 18: 255, 25: 255, 26: 255, 28: 255, 31: 255, 32: 255,
                                33: 255, 34: 255, 35: 255, 36: 255, 37: 255, 38: 255, 39: 255,40: 255, 41: 255, 42: 255, 43: 255, 44: 255,
                                46: 255, 51: 255, 53: 255, 56: 255, 59: 255, 60: 255, 62: 255, 63: 255, 64: 255, 65: 255, }
            id_to_instanceid = {19:1, 20:2, 21:2, 22:2, 55:3, 61:4, 54:5, 58:6, 57:7, 52:8}
        gt = dict()
        gt['images'] = stff_json['images']
        gt['categories'] = [{'id': 1, 'name': 'person'},
        {'id': 2, 'name': 'rider'},
        {'id': 3, 'name': 'car'},
        {'id': 4, 'name': 'truck'},
        {'id': 5, 'name': 'bus'},
        {'id': 6, 'name': 'train'},
        {'id': 7, 'name': 'motorcycle'},
        {'id': 8, 'name': 'bicycle'}]
'''

def get_map_m2c():
    map_mapillary_to_cityscapes = {
    # mapillary : cityscapes
        # road
        7: 7,
        8: 7,
        10:7,
        13:7,
        14:7,
        23:7,
        24:7,
        # sidewalk
        2:8,
        9:8,
        11:8,
        15:8,
        # building
        17: 11,
        # wall
        6: 12,
        # fence
        3: 13,
        # pole
        45: 17,
        47: 17,
        # traffic light
        48: 19,
        # traffic sign
        49: 20,
        50: 20,
        # vegetation
        30: 21,
        # terrain  TODO: NEW
        29: 22,
        # sky
        27: 23,
        # person / Pedestrian  # THING
        19: 24,
        # rider # THING
        20: 25,
        21: 25,
        22: 25,
        # car                   # THING
        55: 26,
        # truck  TODO: NEW
        61: 27,
        # bus                  # THING
        54: 28,
        # On Rails (in Maipllary) train (in Cityscapes), TODO: NEW
        58: 31,
        # motorcycle           # THING
        57: 32,
        # bicycle              # THING
        52: 33,

    }
    return map_mapillary_to_cityscapes
