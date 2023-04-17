# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHeadPanoptic, ASPPModule


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)




class DepthwiseSeparableASPPHeadMainBlock(ASPPHeadPanoptic):
    """
    Encoder-Decoder with Atrous Separable Convolution for Semantic Image
     Segmentation.

     This head is the implementation of `DeepLabV3+
     <https://arxiv.org/abs/1802.02611>`_.

     Args:
         c1_in_channels (int): The input channels of c1 decoder. If is 0,
             the no decoder will be used.
         c1_channels (int): The intermediate channels of c1 decoder.
         
    This is same as the original DepthwiseSeparableASPPHead
    I have modifed the original DepthwiseSeparableASPPHead to adapt for panoptic segmentation
    The only diff. is that I remove the following line from the forward function to just return the raw feature map
    # output = self.cls_seg(output)
    I use this class to create semantic and instance heads for panoptic segmentation
    This class is used in the DepthwiseSeparableASPPHeadPanoptic class
     """
    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHeadMainBlock, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        # output = self.cls_seg(output)
        return output



@HEADS.register_module()
class DepthwiseSeparableASPPHeadPanoptic(ASPPHeadPanoptic):
    """
    I have created this class for panoptic segmentation
    """
    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHeadPanoptic, self).__init__(**kwargs)
        # self.debug = debug
        self.semanitc_head = DepthwiseSeparableASPPHeadMainBlock(c1_in_channels, c1_channels, **kwargs)
        # self.act_panop = activate_panoptic
        if self.act_panop:
            self.instance_head = DepthwiseSeparableASPPHeadMainBlock(c1_in_channels, c1_channels, **kwargs)
        self.debug_output = {}

    def forward(self, inputs):
        semantic_pred = self.semanitc_head(inputs)
        semantic_pred = self.cls_seg(semantic_pred)
        self.debug_output.update({'semantic': semantic_pred.detach()})
        if self.act_panop:
            x_instance = self.instance_head(inputs)
            center_pred = self.reg_cnt(x_instance)
            offset_pred = self.reg_ofs(x_instance)
            depth_pred = torch.zeros(1).cuda()
            self.debug_output.update({'center': center_pred.detach()})
            self.debug_output.update({'offset': offset_pred.detach()})
            self.debug_output.update({'depth': depth_pred.detach()})
            return semantic_pred, center_pred, offset_pred, depth_pred
        else:
            return semantic_pred


