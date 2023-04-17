# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BaseDecodeHeadPanoptic(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHeadPanoptic.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 debug,
                 activate_panoptic,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg')),
                 freeze_semantic_head=False,
                 train_instance_head_from_scratch=False,
                 ):
        super(BaseDecodeHeadPanoptic, self).__init__(init_cfg)

        self._init_inputs(in_channels, in_index, input_transform)

        self.debug = debug
        self.act_panop = activate_panoptic # TODO
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)

        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if self.act_panop:
            self.conv_center = nn.Conv2d(channels, 1, kernel_size=1)
            self.conv_offset = nn.Conv2d(channels, 2, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      gt_center,
                      center_weights,
                      gt_offset,
                      offset_weights,
                      gt_instance_seg,
                      gt_depth_map,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if not self.act_panop:
            semantic_logits = self.forward(inputs)
            losses = self.losses(semantic_logits, gt_semantic_seg, seg_weight)
        else:
            if img_metas:
                if img_metas[0] == 'mixed_img':
                    loss_idx = [3, 4]
                else:
                    loss_idx = [1, 2]

                semantic_logits, center_logits, offset_logits, depth_logits = self.forward(inputs)
                losses = self.losses_panoptic(loss_idx, semantic_logits, gt_semantic_seg, seg_weight,
                                              center_logits=center_logits, offset_logits=offset_logits, depth_logits=depth_logits,
                                              gt_center=gt_center, center_weights=center_weights, gt_offset=gt_offset, offset_weights=offset_weights,
                                              gt_instance_seg=gt_instance_seg, gt_depth_map=gt_depth_map
                                              )
            else:
                semantic_logits, center_logits, offset_logits, depth_logits = self.forward(inputs)
                losses = self.losses(semantic_logits, gt_semantic_seg, seg_weight)
        return losses

    @force_fp32(apply_to=('seg_logit',))
    def losses_panoptic(self, loss_idx, seg_logit, seg_label, seg_weight=None, **kwargs):
        cnt_loss_idx = loss_idx[0]
        ofs_loss_idx = loss_idx[1]

        """Compute semantic loss."""
        loss = dict()
        seg_logit = resize(input=seg_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode[0](seg_logit, seg_label, weight=seg_weight, ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        """Compute center loss."""
        center_logits = kwargs['center_logits']
        gt_center = kwargs['gt_center']
        center_weights = kwargs['center_weights']
        center_weights = center_weights.squeeze(dim=1)
        center_logits = resize(input=center_logits, size=seg_label.shape[1:], mode='bilinear', align_corners=self.align_corners)
        center_weights = center_weights[:, None, :, :].expand_as(center_logits)
        avg_factor = center_weights.sum()
        if avg_factor > 0:
            loss['loss_center'] = self.loss_decode[cnt_loss_idx](center_logits, gt_center.squeeze(dim=1), weight=center_weights, avg_factor=avg_factor)
        elif avg_factor == 0:
            avg_factor = 1
            loss['loss_center'] = self.loss_decode[cnt_loss_idx](center_logits, gt_center.squeeze(dim=1), weight=center_weights, avg_factor=avg_factor)
        else:
            raise NotImplementedError('avg_factor can not be less than 0 !!')

        """Compute offset loss."""
        offset_logits = kwargs['offset_logits']
        gt_offset = kwargs['gt_offset']
        offset_weights = kwargs['offset_weights']
        offset_weights = offset_weights.squeeze(dim=1)
        offset_logits = resize(input=offset_logits, size=seg_label.shape[1:], mode='bilinear', align_corners=self.align_corners)
        offset_weights = offset_weights[:, None, :, :].expand_as(offset_logits)
        avg_factor = offset_weights.sum()
        if avg_factor > 0:
            loss['loss_offset'] = self.loss_decode[ofs_loss_idx](offset_logits, gt_offset.squeeze(dim=1), weight=offset_weights, avg_factor=avg_factor)
        elif avg_factor == 0:
            avg_factor = 1
            loss['loss_offset'] = self.loss_decode[ofs_loss_idx](offset_logits, gt_offset.squeeze(dim=1), weight=offset_weights, avg_factor=avg_factor)
        else:
            raise NotImplementedError('avg_factor can not be less than 0 !!')

        return loss

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, seg_weight=None, **kwargs):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode[0](
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        return loss

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def reg_cnt(self, feat):
        """ center regression """
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_center(feat)
        return output

    def reg_ofs(self, feat):
        """ offset regression """
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_offset(feat)
        return output