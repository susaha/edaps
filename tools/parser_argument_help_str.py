# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

paht1 = 'override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.'
paht2 = 'override some settings in the used config, the key-value pair ' \
        'in xxx=yyy format will be merged into config file. If the value to ' \
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b ' \
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" ' \
        'Note that the quotation marks are necessary and that no white space ' \
        'is allowed.'
wm1 = 'Can not find "auto_scale_lr" or "auto_scale_lr.enable" or "auto_scale_lr.base_batch_size" in your configuration file. Please update all the configuration files to mmdet >= 2.24.1.'
wm2 = '`--gpu-ids` is deprecated, please use `--gpu-id`. Because we only support single GPU mode in non-distributed training. Use the first GPU in `gpu_ids` now.'

# the below are for tools/test_mmdet.py
str1 =  'override some settings in the used config, the key-value pair ' \
        'in xxx=yyy format will be merged into config file. If the value to ' \
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b ' \
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" ' \
        'Note that the quotation marks are necessary and that no white space ' \
        'is allowed.'
str2 = 'custom options for evaluation, the key-value pair in xxx=yyy ' \
        'format will be kwargs for dataset.evaluate() function (deprecate), ' \
        'change to --eval-options instead.'

str3 = 'custom options for evaluation, the key-value pair in xxx=yyy ' \
        'format will be kwargs for dataset.evaluate() function'
str4 =  '--options and --eval-options cannot be both ' \
        'specified, --options is deprecated in favor of --eval-options'

str5 = ( 'Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
str6 = ('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')