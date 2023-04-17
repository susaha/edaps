# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)


@DETECTORS.register_module()
class MaskRCNNCustomized(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskRCNNCustomized, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):

        # # visulize the gt labels
        # from mmdet.models.utils.visualize_gt_labels import visualize_mask_rcnn_gt_labels
        # visualize_mask_rcnn_gt_labels(img, img_metas, gt_bboxes, gt_labels, gt_masks, gt_bboxes_ignore)
        # print('gt_labels')
        # print(gt_labels)
        # print('gt_bboxes')
        # print(gt_bboxes)

        log_vars = {}
        losses = {}
        batch_size = len(gt_labels)
        set_loss_to_zero = False
        for i in range(batch_size):
            if gt_labels[i].numel() == 0:
                set_loss_to_zero = True
                break
        if not set_loss_to_zero:
            x = self.extract_feat(img)
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train( x, img_metas, gt_bboxes,  gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg, **kwargs)
                losses.update(rpn_losses)
                # rpn_losses, rpn_log_vars = self._parse_losses(rpn_losses)
                # log_vars.update(rpn_log_vars)
            else:
                proposal_list = proposals
            # RoIHead (includes box and mask head) forward and loss
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks,  **kwargs)
            losses.update(roi_losses)
            # roi_losses, roi_log_vars = self._parse_losses(roi_losses)
            # log_vars.update(roi_log_vars)
            # loss = rpn_losses + roi_losses
            loss, clean_log_vars = self._parse_losses(losses)
            loss.backward()
            log_vars.update(clean_log_vars)
            # log_vars['loss'] = rpn_log_vars['loss'] + roi_log_vars['loss']
        return log_vars

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        log_vars = self(**data)
        optimizer.step()
        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars, num_samples=len(data['img_metas']))
        # outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
