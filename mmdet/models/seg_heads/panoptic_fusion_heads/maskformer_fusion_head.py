# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.mask import mask2bbox
from mmdet.models.builder import HEADS
from .base_panoptic_fusion_head import BasePanopticFusionHead


@HEADS.register_module()
class MaskFormerFusionHead(BasePanopticFusionHead):

    def __init__(self,
                 num_things_classes=8,
                 num_stuff_classes=11,
                 cityscapes_thing_list=[11, 12, 13, 14, 15, 16, 17, 18],
                 cityscapes_label_divisor=1000,
                 semantic_inference_type='semantic_inference',
                 test_cfg=None,
                 loss_panoptic=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(num_things_classes, num_stuff_classes, test_cfg,
                         loss_panoptic, init_cfg, **kwargs)
        self.cityscapes_thing_list = cityscapes_thing_list
        self.semantic_inference_type = semantic_inference_type
        self.cityscapes_label_divisor = cityscapes_label_divisor

    def forward_train(self, **kwargs):
        """MaskFormerFusionHead has no training loss."""
        return dict()

    def panoptic_postprocess(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr) # exculding the background class prediction, if there are 19 classes then labels are from 0 to 18,
                                                                        # and the 19th label-id is for no-calss or background class,
                                                                        # MaskFormer mask classification layer is trained on total_num_classes + 1 classes
                                                                        # i.e. for cityscapes, it predicts [N x K] mask classification scores,
                                                                        # N:num_queries=100, K=self.num_classes+1=20
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w), 255 * self.cityscapes_label_divisor, dtype=torch.int32, device=cur_masks.device)
        # panoptic_seg = torch.full((h, w), self.num_classes, dtype=torch.int32, device=cur_masks.device) # original
        # semantic_seg = torch.full((self.num_classes, h, w), self.num_classes, dtype=torch.int32, device=cur_masks.device) # added by Suman
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())

                # isthing = pred_class < self.num_things_classes # original
                if pred_class in self.cityscapes_thing_list: # this if-else block is added by Suman
                    isthing = True
                else:
                    isthing = False

                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if filter_low_score:
                    mask = mask & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class * self.cityscapes_label_divisor
                    else:
                        panoptic_seg[mask] = (pred_class * self.cityscapes_label_divisor + instance_id)
                        # panoptic_seg[mask] = (pred_class + instance_id * INSTANCE_OFFSET) # original used for COCO
                        #  class_id * label_divisor + new_ins_id
                        instance_id += 1
                    # semantic_seg[pred_class, mask] = pred_class  # added by Suman

        return panoptic_seg

    def semantic_postprocess(self, mask_cls, mask_pred):
        """Semantic segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Semantic segment result of shape \
                (cls_out_channels, h, w).
        """
        # K: num classes, N: num queries, H: hieght, W: width
        num_query, num_classes = mask_cls.shape  # [N, K]
        num_classes = num_classes - 1
        mask_cls = mask_cls[:, :num_classes]
        _, imgH, imgW = mask_pred.shape  # [N, H, W]
        mask_cls = F.softmax(mask_cls, dim=-1)  # convert logits to softmax prob., softamx over K classes
        mask_cls = mask_cls.permute(1, 0)  # [N, K] -> [K, N] : permute matrix for matrix multiplication
        mask_pred = mask_pred.sigmoid()  # converts logits to score between 0 and 1
        mask_pred = mask_pred.view(num_query, -1)  # [N, H, W] -> [N, H*W] : change view for matrix multiplication
        semantic_seg = torch.matmul(mask_cls, mask_pred)  # [K, N] x [N, H*W] --> [K, H*W] # during this matrix multiplication, we multiply sigmoid and softmax prob scores, but
                                                          # but the output matrix might have values > 1
        semantic_seg = semantic_seg.view(num_classes, imgH, imgW)  # [K, H*W] --> [K, H, W]
        # semseg = semantic_seg.argmax(0) # final prediction
        # --- visualization ---
        # rows, cols = 2, 2
        # gridspec_kw = {'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        # fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw=gridspec_kw)
        # semseg = semantic_seg.argmax(0)
        # subplotimg(axs[0][0], semseg, 'SemGT', cmap='cityscapes')
        # for ax in axs.flat:
        #     ax.axis('off')
        # out_dir = '/home/suman/Downloads'
        # outfile = os.path.join(out_dir, meta['ori_filename'])
        # plt.savefig(outfile)
        # plt.close()
        return semantic_seg


    def instance_postprocess(self, mask_cls, mask_pred, pseudo_threshold_instance=0):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels_per_image (Tensor): Predicted labels,\
                shape (n, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (n, h, w).
        """
        max_per_image = self.test_cfg.get('max_per_image', 100) # max_per_image = 100
        num_queries = mask_cls.shape[0]                         # mask_cls.shape = torch.Size([100, 20])
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]            # scores.shape = torch.Size([100, 19]); :-1 to exclude the "no-object" class
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1) # labels.shape = 1900, labels has values [0,1,2,..,18, 0,1,2,...,18, 0,1,2,..18] i.e. repating for 100 queries
        scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False) # scores.flatten(0, 1).shape = 1900; scores_per_image.shape = 100, top_indices.shape=100, selecting the 100 queries (there scores and indices) with top scores
        labels_per_image = labels[top_indices] # labels_per_image.shape=100, labels for topk 100 queries

        query_indices = top_indices // self.num_classes # top_indices value ranges between 0 and 1900, map them to query indices, i.t., 0 to 99, for selecting the best masks, top_indices.shape = 100, self.num_classes=19, query_indices.shape=100,
        mask_pred = mask_pred[query_indices] # query_indices value ranges between 0 and 99, query_indices.shape=100, mask_pred.shape=torch.Size([100, 512, 1024])

        # extract things
        is_thing = []
        for c in labels_per_image:
            if c.item() in self.cityscapes_thing_list:
                is_thing.append(True)
            else:
                is_thing.append(False)
        is_thing = torch.BoolTensor(is_thing) # is_thing.shape=100, bool tensor to pick the scores, labels and masks

        scores_per_image = scores_per_image[is_thing] # before: scores_per_image.shape=100, after: scores_per_image.shape=57
        labels_per_image = labels_per_image[is_thing] # before: labels_per_image.shape=100, after: labels_per_image.shape=57
        mask_pred = mask_pred[is_thing]               # before: mask_pred.shape=torch.Size([100, 512, 1024]), after: mask_pred.shape=torch.Size([57, 512, 1024])

        mask_pred_binary = (mask_pred > 0).float()  # mask_pred.min()=-477.0528, mask_pred.max()=22.03, raw logits, only cosider those pixels as valid predictions which have predicted logits > 0, convert the bool to float, values 0 or 1
        mask_scores_per_image = (mask_pred.sigmoid() *  mask_pred_binary).flatten(1).sum(1) / ( mask_pred_binary.flatten(1).sum(1) + 1e-6) # mask_pred.sigmoid().shape=torch.Size([57, 512, 1024]), (mask_pred.sigmoid() *  mask_pred_binary).flatten(1).shape=torch.Size([57, 524288]), (mask_pred.sigmoid() *  mask_pred_binary).flatten(1).sum(1).shape=57, mask_pred_binary.flatten(1).shape=torch.Size([57, 524288]), mask_pred_binary.flatten(1).sum(1).shape=57, mask_scores_per_image.shape=57,tensor([0.9687, 0.8592, 0.8592, 0.8305, 0.0000, 0..., 0.8631], device='cuda:0')
        det_scores = scores_per_image * mask_scores_per_image # scores_per_image.shape=mask_scores_per_image.shape=det_scores.shape=57, scores_per_image=mask_scores_per_image=det_scores=[value between 0 and 1,]
        mask_pred_binary = mask_pred_binary.bool() # mask_pred_binary.shape=torch.Size([57, 512, 1024]), convert float to bool

        if pseudo_threshold_instance > 0:
            # labels_per_image = labels_per_image[det_scores > pseudo_threshold_instance]
            # bboxes = bboxes[det_scores > pseudo_threshold_instance]
            # mask_pred_binary = mask_pred_binary[det_scores > pseudo_threshold_instance]
            mask_pseudo_weight = torch.sum(det_scores > pseudo_threshold_instance).item() / det_scores.shape[0]
            return labels_per_image, mask_pred_binary, mask_pseudo_weight
        else:
            bboxes = mask2bbox(mask_pred_binary)
            bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)  # bboxes.shape=torch.Size([57, 5])
            return labels_per_image, bboxes, mask_pred_binary

    def simple_test(self,
                    mask_cls_results,
                    mask_pred_results,
                    img_metas,
                    rescale=False,
                    pseudo_threshold_instance=0,
                    **kwargs):

        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        # assert not semantic_on, 'segmantic segmentation  results are not supported yet.'

        results = []
        for mask_cls_result, mask_pred_result, meta in zip(mask_cls_results, mask_pred_results, img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(mask_pred_result[:, None], size=(ori_height, ori_width),  mode='bilinear', align_corners=False)[:, 0]
            else:
                # this else part is for debug mode and not for actual evaluation, for actual evaluation, we do on original shape
                mask_pred_result = F.interpolate(mask_pred_result[:, None], size=(512, 1024), mode='bilinear', align_corners=False)[:, 0]

            result = dict()
            if panoptic_on and pseudo_threshold_instance == 0: # because for pseduo labels we dont need panoptic predictions
                pan_results = self.panoptic_postprocess(mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results

            if instance_on:
                ins_results = self.instance_postprocess(mask_cls_result, mask_pred_result, pseudo_threshold_instance=pseudo_threshold_instance)
                result['ins_results'] = ins_results

            if semantic_on:
                # MaskFormer paper propose two ways of semantic inference: (1) general inference and (2) smenaitc inference under Sec. 3.4 Mask-classification inference
                # As per the authors of MaskFormer: for semantic segmentation evaluation, semantic inference gives better results
                assert self.semantic_inference_type == 'semantic_inference', 'semantic_inference_type must be set to "semantic_inference", and not to ' \
                                                        '"general_inference". The general inference is not implemented yet!'
                if self.semantic_inference_type == 'semantic_inference':
                    sem_results = self.semantic_postprocess(mask_cls_result, mask_pred_result)
                    result['sem_results'] = sem_results

            results.append(result)

        return results
