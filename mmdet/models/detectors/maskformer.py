# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np

from mmdet.core import INSTANCE_OFFSET, bbox2result, bbox2resultCityscapes
from mmdet.core.visualization import imshow_det_bboxes
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
#
# the following imorts are added by Suman on Aug. 30 2022, for _visualize_gt_labels()
from mmdet.models.utils.visualize_gt_labels import subplotimg, get_mean_std, denorm, random_color, subplotInstSeg
from matplotlib import pyplot as plt
import os
import torch
import torch.nn.functional as F
from mmseg.core import add_prefix

@DETECTORS.register_module()
class MaskFormer(SingleStageDetector):
    r"""Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 ):
        super(SingleStageDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        panoptic_head_ = copy.deepcopy(panoptic_head)
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = build_head(panoptic_head_)

        panoptic_fusion_head_ = copy.deepcopy(panoptic_fusion_head)
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = build_head(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes
        self.cityscapes_thing_list = self.panoptic_head.cityscapes_thing_list

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # BaseDetector.show_result default for instance segmentation
        if self.num_stuff_classes > 0:
            self.show_result = self._show_pan_result
        self.debug_output = {}

    def forward_dummy(self, img, img_metas):
        """Used for computing network flops. See
        `mmdetection/tools/analysis_tools/get_flops.py`

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        outs = self.panoptic_head(x, img_metas)
        return outs

    def encode_decode(self, img, img_metas, pseudo_threshold_instance=0.968):
        assert pseudo_threshold_instance > 0, 'for pseudo label generation the pseudo_threshold_instance value must be > 0!'
        '''
        post_process_results['semantic']
        post_process_results['masks']
        post_process_results['labels']
        '''
        # add batch_input_shape in img_metas
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)  # backbone forward pass
        # forward
        all_cls_scores, all_mask_preds = self.panoptic_head(x, img_metas)
        post_process_results = self._get_preds_for_pseudo_labels(all_cls_scores, all_mask_preds, img_metas, pseudo_threshold_instance=pseudo_threshold_instance)
        pp_semantic = post_process_results['semantic']
        pp_masks = post_process_results['masks']
        pp_masks_lables = post_process_results['labels']
        mask_pseudo_weight = post_process_results['mask_pseudo_weight']

        return pp_semantic, pp_masks, pp_masks_lables, mask_pseudo_weight

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,  gt_masks, gt_semantic_seg=None,  gt_bboxes_ignore=None, return_feat=False, update_output_debug=False, **kargs):

        # visulize the gt labels
        # self._visualize_gt_labels(img, img_metas, gt_bboxes, gt_labels, gt_masks, gt_semantic_seg, gt_bboxes_ignore)

        # add batch_input_shape in img_metas
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img) # backbone forward pass
        losses = dict()
        if return_feat:
            losses['features'] = x # saving the features for computing feat. distance loss
        maskformer_losses = self.panoptic_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_masks,  gt_semantic_seg,  gt_bboxes_ignore, update_output_debug=update_output_debug)

        if update_output_debug:
            all_cls_scores = maskformer_losses.pop('all_cls_scores')
            all_mask_preds = maskformer_losses.pop('all_mask_preds')
            losses.update(add_prefix(maskformer_losses, 'mf'))
            # update the output debug for train time visualization
            self._get_preds_for_pseudo_labels(all_cls_scores, all_mask_preds, img_metas, pseudo_threshold_instance=0.0)
        else:
            losses.update(maskformer_losses)
        return losses

    def _get_preds_for_pseudo_labels(self, all_cls_scores, all_mask_preds, img_metas, pseudo_threshold_instance=0.0):
        '''
        all_cls_scores.shape=torch.Size([6, 1, 100, 20]) # TODO: 6 decoder laeyrs output, 100 queries, 134 classes
        all_mask_preds.shape=torch.Size([6, 1, 100, 60, 90]) # TODO: 6 decoder laeyrs output, 100 queries, spatial dim. (HxW) of mask prediction feautre map

        Instance prediction format explained below:
            results is a list : len(results) = number of images in a mini batch
            labels_per_image, bboxes, mask_pred_binary = results[i]['ins_results']
            labels_per_image is a tensor of shape N where N is the topK predicted masks, each entry is a class label for a predited mask
            mask_pred_binary: is a tensor of shape (N, H, W)
        returned value:
            results[i]['sem_results']
            results[i]['ins_results']
        '''
        with torch.no_grad():
            mask_cls_results = all_cls_scores[-1] # selecting the last layer predictions: (torch.Size([1, 100, 20]))
            mask_pred_results = all_mask_preds[-1] # selecting the last layer predictions (torch.Size([1, 100, 60, 90]))
            # upsample masks
            img_shape = img_metas[0]['batch_input_shape']
            mask_pred_results = F.interpolate(mask_pred_results, size=(img_shape[0], img_shape[1]), mode='bilinear', align_corners=False)
            kwargs = dict(rescale=False)
            # postprocessing of results
            results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas,
                                                            pseudo_threshold_instance=pseudo_threshold_instance, **kwargs) # TODO
            for i in range(len(results)):
                if 'sem_results' in results[i]:
                    results[i]['sem_results'] = F.softmax(results[i]['sem_results'], dim=0)
                    self.debug_output.update({'semantic': results[i]['sem_results'].detach()})
                    # self.debug_output.update({'semantic': results[i]['sem_results'].detach().clone()})
                if 'ins_results' in results[i]:
                    labels_per_image, mask_pred_binary, mask_pseudo_weight = results[i]['ins_results']
                    self.debug_output.update({'masks': mask_pred_binary.detach()})
                    self.debug_output.update({'labels': labels_per_image.detach()})
                    if pseudo_threshold_instance > 0:
                        self.debug_output.update({'mask_pseudo_weight': mask_pseudo_weight})
                    # self.debug_output.update({'masks': mask_pred_binary.detach().clone()})
                    # self.debug_output.update({'labels': labels_per_image.detach().clone()})

        if pseudo_threshold_instance > 0:
            return results

    def _visualize_gt_labels(self, img, img_metas, gt_bboxes, gt_labels, gt_masks, gt_semantic_seg, gt_bboxes_ignore):
        '''
        gt_booxes and gt_labels are avaible only for thing classes
        gt_masks are avaibale for both thing and stuff classes
        '''

        means, stds = get_mean_std(img_metas, 'cuda:0')
        vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
        out_dir = '/home/suman/Downloads'
        batch_size = 2
        for bid in range(batch_size):
            rows, cols = 2, 2
            gridspec_kw = {'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
            fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw=gridspec_kw)
            subplotimg(axs[1][1], vis_img[bid], 'BoxGT', drawboxes=True, bboxes=gt_bboxes[bid])
            subplotimg(axs[0][0], vis_img[bid], 'Img')
            subplotimg(axs[0][1], gt_semantic_seg[bid], 'SemGT', cmap='cityscapes')
            subplotimg(axs[1][0], subplotInstSeg(vis_img[bid], gt_masks[bid].masks), 'InstGT')

            # gt boxes
            for ax in axs.flat:
                ax.axis('off')
            dname = 'synthia'
            if dname == 'COCO':
                outfile = os.path.join(out_dir, img_metas[0]['ori_filename'])
            elif dname == 'cityscapes':
                str1 = img_metas[0]['ori_filename']
                str2 = str1.split('/')[0]
                str3 = os.path.join(out_dir, str2)
                os.makedirs(str3, exist_ok=True)
                outfile = os.path.join(out_dir, img_metas[0]['ori_filename'])
            elif dname == 'synthia':
                os.makedirs(os.path.join(out_dir, 'synthia_gt_visual', f'bid-{bid}'), exist_ok=True)
                outfile = os.path.join(out_dir, 'synthia_gt_visual', f'bid-{bid}', img_metas[0]['ori_filename'])
            plt.savefig(outfile)
            plt.close()
            # print()

    def simple_test(self, imgs, img_metas, **kwargs):
        """Test without augmentation.

        Args:
            imgs (Tensor): A batch of images.
            img_metas (list[dict]): List of image information.

        Returns:
            list[dict[str, np.array | tuple[list]] | tuple[list]]:
                Semantic segmentation results and panoptic segmentation \
                results of each image for panoptic segmentation, or formatted \
                bbox and mask results of each image for instance segmentation.

            .. code-block:: none

                [
                    # panoptic segmentation
                    {
                        'pan_results': np.array, # shape = [h, w]
                        'ins_results': tuple[list],
                        # semantic segmentation results are not supported yet
                        'sem_results': np.array
                    },
                    ...
                ]

            or

            .. code-block:: none

                [
                    # instance segmentation
                    (
                        bboxes, # list[np.array]
                        masks # list[list[np.array]]
                    ),
                    ...
                ]
        """
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)
        for i in range(len(results)):
            if 'pan_results' in results[i]:
                results[i]['pan_results'] = results[i]['pan_results'].detach().cpu().numpy()
            if 'sem_results' in results[i]:
                results[i]['sem_results'] = results[i]['sem_results'].detach().cpu().numpy()

            if 'ins_results' in results[i]:
                labels_per_image, bboxes, mask_pred_binary = results[i]['ins_results']

                bbox_results = bbox2resultCityscapes(bboxes, labels_per_image, self.num_classes, self.cityscapes_thing_list) # added by Suman
                # bbox_results = bbox2result(bboxes, labels_per_image, self.num_things_classes) # Original

                mask_results = [[] for _ in range(self.num_classes)] # added by suman
                # mask_results = [[] for _ in range(self.num_things_classes)] # original
                for j, label in enumerate(labels_per_image):
                    mask = mask_pred_binary[j].detach().cpu().numpy()
                    mask_results[label].append(mask)
                results[i]['ins_results'] = bbox_results, mask_results

            # assert 'sem_results' not in results[i], 'segmantic segmentation results are not supported yet.'

        if self.num_stuff_classes == 0:
            results = [res['ins_results'] for res in results]

        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def onnx_export(self, img, img_metas):
        raise NotImplementedError

    def _show_pan_result(self,
                         img,
                         result,
                         score_thr=0.3,
                         bbox_color=(72, 101, 241),
                         text_color=(72, 101, 241),
                         mask_color=None,
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=False,
                         wait_time=0,
                         out_file=None):
        """Draw `panoptic result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """


        img = mmcv.imread(img)
        img = img.copy()
        pan_results = result['pan_results']
        # keep objects ahead
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != self.num_classes  # for VOID label
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (pan_results[None] == ids[:, None, None])

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            segms=segms,
            labels=labels,
            class_names=self.CLASSES,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
