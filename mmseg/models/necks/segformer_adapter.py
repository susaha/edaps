import torch
import torch.nn as nn

from mmseg.ops import resize
from ..builder import NECKS


@NECKS.register_module()
class SegFormerAdapter(nn.Module):

    '''
    As the MiT encoder of SegFormer has an output stride of 32
    but the DeepLabV3+ decoder is designed for an output stride of 8,
    we bilinearly upsample the SegFormer bottleneck features by ×4
    when combined with the DeepLabv3+ decoder
    '''

    def __init__(self, out_layers=[3], scales=[4]):
        super(SegFormerAdapter, self).__init__()
        self.out_layers = out_layers
        self.scales = scales

    def forward(self, x):
        _c = {}
        for i, s in zip(self.out_layers, self.scales):
            if s == 1:
                _c[i] = x[i]
            else:
                _c[i] = resize(
                    x[i], scale_factor=s, mode='bilinear', align_corners=False)
            # mmcv.print_log(f'{i}: {x[i].shape}, {_c[i].shape}', 'mmseg')

        x[-1] = torch.cat(list(_c.values()), dim=1)
        return x
