# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head_panoptic import BaseDecodeHeadPanoptic
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


def get_layers(in_index, in_channels, embed_dims, channels, embed_neck_cfg, embed_cfg, fusion_cfg):
    embed_layers = {}
    for i, in_channels, embed_dim in zip(in_index, in_channels, embed_dims):
        if i == in_index[-1]:
            embed_layers[str(i)] = build_layer(in_channels, embed_dim, **embed_neck_cfg)
        else:
            embed_layers[str(i)] = build_layer(in_channels, embed_dim, **embed_cfg)
    embed_layers = nn.ModuleDict(embed_layers)
    fuse_layer = build_layer(sum(embed_dims), channels, **fusion_cfg)
    return embed_layers, fuse_layer


def head_forward(inputs, in_index, embed_layers, fuse_layer, align_corners):
    x = inputs
    n, _, h, w = x[-1].shape
    os_size = x[0].size()[2:]
    _c = {}
    for i in in_index:
        _c[i] = embed_layers[str(i)](x[i])
        if _c[i].dim() == 3:
            _c[i] = _c[i].permute(0, 2, 1).contiguous().reshape(n, -1, x[i].shape[2], x[i].shape[3])
        if _c[i].size()[2:] != os_size:
            _c[i] = resize(_c[i], size=os_size, mode='bilinear', align_corners=align_corners)
    return fuse_layer(torch.cat(list(_c.values()), dim=1))


@HEADS.register_module()
class DAFormerHeadPanoptic(BaseDecodeHeadPanoptic):
    def __init__(self, **kwargs):
        super(DAFormerHeadPanoptic, self).__init__(input_transform='multiple_select', **kwargs)
        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners
        # semantic head
        self.embed_layers_semantic, self.fuse_layer_semantic = get_layers(self.in_index, self.in_channels, embed_dims, self.channels, embed_neck_cfg, embed_cfg, fusion_cfg)
        self.act_panop = kwargs['activate_panoptic']
        if self.act_panop:
            # instance head
            self.embed_layers_instance, self.fuse_layer_instance = get_layers(self.in_index, self.in_channels, embed_dims, self.channels, embed_neck_cfg, embed_cfg, fusion_cfg)
        self.debug = kwargs['debug']
        self.debug_output = {}

    def forward(self, inputs):
        # semanitc head forward pass
        semantic_pred = head_forward(inputs, self.in_index, self.embed_layers_semantic, self.fuse_layer_semantic, self.align_corners)
        semantic_pred = self.cls_seg(semantic_pred)
        self.debug_output.update({'semantic': semantic_pred.detach()})
        if self.act_panop:
            x_instance = head_forward(inputs, self.in_index, self.embed_layers_instance, self.fuse_layer_instance, self.align_corners)
            center_pred = self.reg_cnt(x_instance)
            offset_pred = self.reg_ofs(x_instance)
            depth_pred = torch.zeros(1).cuda()
            self.debug_output.update({'center': center_pred.detach()})
            self.debug_output.update({'offset': offset_pred.detach()})
            self.debug_output.update({'depth': depth_pred.detach()})
            return semantic_pred, center_pred, offset_pred, depth_pred
        else:
            return semantic_pred


@HEADS.register_module()
class DAFormerHeadPanopticShared(BaseDecodeHeadPanoptic):
    def __init__(self, **kwargs):
        super(DAFormerHeadPanopticShared, self).__init__(input_transform='multiple_select', **kwargs)
        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners
        # semantic head
        self.embed_layers_semantic, self.fuse_layer_semantic = get_layers(self.in_index, self.in_channels, embed_dims, self.channels, embed_neck_cfg, embed_cfg, fusion_cfg)
        self.act_panop = kwargs['activate_panoptic']
        self.debug = kwargs['debug']
        self.debug_output = {}

    def forward(self, inputs):
        # semanitc head forward pass
        x = head_forward(inputs, self.in_index, self.embed_layers_semantic, self.fuse_layer_semantic, self.align_corners)
        semantic_pred = self.cls_seg(x)
        self.debug_output.update({'semantic': semantic_pred.detach()})
        center_pred = self.reg_cnt(x)
        offset_pred = self.reg_ofs(x)
        self.debug_output.update({'center': center_pred.detach()})
        self.debug_output.update({'offset': offset_pred.detach()})
        depth_pred = torch.zeros(1).cuda()
        self.debug_output.update({'depth': depth_pred.detach()})
        return semantic_pred, center_pred, offset_pred, depth_pred