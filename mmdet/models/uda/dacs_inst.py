# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd


from mmseg.core import add_prefix
from mmdet.models import UDA, build_detector
from mmdet.models.uda.uda_decorator import UDADecorator, get_module
from mmdet.models.utils.dacs_transforms import (denorm, get_class_masks, get_class_masks_dummy, get_class_masks_only_thing, get_mean_std, strong_transform)
from mmseg.utils.visualize_pred  import subplotimg, convertNDMaskTo2DMask, convertNDMaskListToNDMaskTensor
from mmdet.utils.utils import downscale_label_ratio
from .dacs import DACS, calc_grad_magnitude
from mmseg.ops import resize
from mmseg.datasets.pipelines.gen_panoptic_labels_for_maskformer import isValidBox, get_bbox_coord
from mmdet.core import BitmapMasks

def get_ema_box_mask_preds_for_visual(out_roi_head, batch_size=2, dev=None, pseudo_threshold=0.9, crop_size=(512,512)):
    masks_batch_list = []
    bboxes_batch_list = []
    for bid in range(batch_size):
        masks_list = []
        bboxes_list = []
        boxes, masks = out_roi_head[bid]
        assert len(boxes) == len(masks), 'boxes and masks lists must have same length!'
        num_instance_classes = len(boxes)
        for c in range(num_instance_classes):
            boxes_c = boxes[c]
            if boxes_c.shape[0] > 0:
                masks_c = masks[c]
                assert boxes_c.shape[0] == len(masks_c), 'there must be same number of masks and boxes for a class!'
                for m in range(len(masks_c)):  # loop over masks of class c
                    mask_score = boxes_c[m, 4]
                    if mask_score >= pseudo_threshold:
                        masks_list.append(torch.from_numpy(masks_c[m]).to(dev).byte())
                        bboxes_list.append(torch.from_numpy(boxes_c[m,:]).to(dev))
        if masks_list:
            masks_list = torch.stack(masks_list, dim=0)
            bboxes_list = torch.stack(bboxes_list, dim=0)
        else:
            ch, cw = crop_size
            masks_list = torch.zeros((1, ch, cw), device=dev).byte()
            bboxes_list = torch.zeros((1, 5), device=dev).float()
            bboxes_list[0][0], bboxes_list[0][1], bboxes_list[0][2], bboxes_list[0][3], bboxes_list[0][4] = 1, 1, ch-1, cw-1, 0.0
        masks_batch_list.append(masks_list)
        bboxes_batch_list.append(bboxes_list)
    return bboxes_batch_list, masks_batch_list


def get_ema_panoptic_label(out_roi_head, pseudo_threshold=0.9, map_pos_index_to_cid=None, pred_shape=None, dev=None,
                           thing_list=None, label_divisor=10000, batch_size=2, src_max_inst_per_class=None,
                           map_cid_to_pos_index=None, inst_pseduo_weight=None):

    assert inst_pseduo_weight is not None, 'inst_pseduo_weight should not have a None value!!'
    pan_seg_list = []
    # pseudo_weight_instance_list = []
    pseudo_wght_val_list = []
    for bid in range(batch_size):
        boxes, masks = out_roi_head[bid]
        assert len(boxes) == len(masks), 'boxes and masks lists must have same length!'
        num_instance_classes = len(boxes)
        pan_seg = np.zeros(pred_shape).astype(int) # if there is no valid mask then all entry is 0 means all stuff regions,
        ins_cnt = 1
        tot_inst_cnt = 1
        class_id_tracker = {}
        instance_count_offset = 1
        for cid in thing_list:
            class_id_tracker[cid] = src_max_inst_per_class[bid][map_cid_to_pos_index[cid]].item() + instance_count_offset
        for c in range(num_instance_classes):
            boxes_c = boxes[c]
            if boxes_c.shape[0] > 0:
                masks_c = masks[c]
                assert boxes_c.shape[0] == len(masks_c), 'there must be same number of masks and boxes for a class!'
                for m in range(len(masks_c)):  # loop over masks of class c
                    mask_score = boxes_c[m, 4]
                    if mask_score >= pseudo_threshold:
                        ins_cnt += 1
                        pan_seg[masks_c[m]] = map_pos_index_to_cid[c] * label_divisor + class_id_tracker[map_pos_index_to_cid[c]]
                        class_id_tracker[map_pos_index_to_cid[c]] += 1
                    tot_inst_cnt += 1
        pan_seg = torch.from_numpy(pan_seg).long()
        pseudo_wght_val = inst_pseduo_weight
        # pseudo_wght_val = (ins_cnt / tot_inst_cnt)
        # pseudo_weight_instance = pseudo_wght_val * torch.ones(pred_shape, device=dev)
        pan_seg_list.append(pan_seg)
        # pseudo_weight_instance_list.append(pseudo_weight_instance)
        pseudo_wght_val_list.append(pseudo_wght_val)
    ema_panoptic_label = torch.stack(pan_seg_list, dim=0).to(dev)
    # pseudo_weight_instance = torch.stack(pseudo_weight_instance_list, dim=0).to(dev)
    # pseudo_wght_val = torch.FloatTensor(pseudo_wght_val_list).to(dev)
    return ema_panoptic_label, pseudo_wght_val_list

def mapId2Domain(id):
    '''
    source panoptic ids are created as: label * 1000 + instance count
    where label = [11,12,...,18]
    So the maximum value could 18 * 1000 + 1000 = 19000 (assuming the maximum no. of instance per class is 1000)
    target panoptic ids are created as: label * 10000 + instance count
    so the minimum valude could be 11 * 10000 + 1 = 110001
    110001 always bigger than 99999
    Return Value:
        this function returns 1 if the panoptic id belongs to target image other wise 0
    '''
    return 1 if id > 99999 else 0 # 1:target domain, 0: source domain

def get_box_mask_pseduo_labels(mixed_lbl_pan, dev, batch_size, map_cid_to_pos_index, label_divisor, label_divisor_target):
    '''
    bboxes_domain_indicator: a tesnor contains 0 for source boxes (and masks), 1 for  target boxes (and masks)
    '''

    bboxes_batch = []
    bboxes_domain_indicator_batch = []
    labels_batch = []
    masks_batch = []
    pseudo_batch_count = 0
    for bid in range(batch_size):
        bboxes = []
        bboxes_domain_indicator = []
        labels = []
        masks = []
        mixed_pan = mixed_lbl_pan[bid][0]
        ids = mixed_pan.unique()
        height, width = mixed_pan.shape
        # print(ids)
        any_valid_box = False
        for id in ids:
            # ignore the stuff segments
            if id == 0:
                continue
            mask = mixed_pan == id
            mask = mask.cpu().numpy()
            box = get_bbox_coord(mask)
            if isValidBox(box):
                any_valid_box = True
                domainId = mapId2Domain(id) # 0: source domain, 1: target domain
                masks.append(mask.astype(np.uint8))
                bboxes.append(torch.FloatTensor(box))
                lbl_divisor = label_divisor if domainId == 0 else label_divisor_target
                cid = int(id / lbl_divisor)
                # print(f'domainId: {domainId}, lbl_divisor: {lbl_divisor}, cid: {cid}')
                labels.append(map_cid_to_pos_index[cid])
                bboxes_domain_indicator.append(domainId) # for each box, it stores either 0 (source domain), or 1 (target domain)
        if not any_valid_box:
            break
        pseudo_batch_count += 1
        bboxes = torch.stack(bboxes, dim=0).to(dev)
        labels = torch.LongTensor(labels).to(dev)
        masks = BitmapMasks(masks, height, width)
        bboxes_domain_indicator = torch.LongTensor(bboxes_domain_indicator).to(dev)
        bboxes_batch.append(bboxes)
        labels_batch.append(labels)
        masks_batch.append(masks)
        bboxes_domain_indicator_batch.append(bboxes_domain_indicator)
    # if one target image out of two, does not have a single valid pseduo box,
    # then there will be no UDA instance loss computed
    if pseudo_batch_count == batch_size:
        return bboxes_batch, labels_batch, masks_batch, bboxes_domain_indicator_batch
    else:
        return None, None, None, None


@UDA.register_module()
class DACSInst(DACS):
    def __init__(self, **cfg):
        super(DACSInst, self).__init__(**cfg)
        self.activate_uda_inst_losses = cfg['activate_uda_inst_losses']
        self.mix_masks_only_thing = cfg['mix_masks_only_thing']
        self.inst_pseduo_weight = cfg['inst_pseduo_weight']
        self.swtich_off_mix_sampling = cfg['swtich_off_mix_sampling']
        self.switch_off_self_training = cfg['switch_off_self_training']

    def getPdSem(self, img):
        debug_output = self.get_model().decode_head.debug_output
        PdSem = debug_output['semantic']
        PdSem = resize(input=PdSem, size=img.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)
        PdSem = torch.softmax(PdSem, dim=1)
        _, PdSem = torch.max(PdSem, dim=1)
        return PdSem

    def getPdBox(self, dev):
        debug_output = self.get_model().roi_head.debug_output
        PdBox = debug_output['pred_bboxes']
        PdLbl = debug_output['pred_labels']
        PdMask = convertNDMaskListToNDMaskTensor(debug_output['pred_masks'], dev=dev)
        return PdBox, PdLbl, PdMask

    def getInstPseudoWeight(self, inst_pseduo_weight=None):
        if inst_pseduo_weight is None:
            return self.local_iter * (1 / self.max_iters)
        else:
            return inst_pseduo_weight

    def forward_train(self, img, img_metas, gt_semantic_seg, gt_bboxes, gt_labels,
                      gt_masks, target_img, target_img_metas,
                      gt_panoptic_only_thing_classes, max_inst_per_class):

        '''
        max_inst_per_class: is a list. The legnth of the list = the batch_siye.
        Each element in a list is a tensor of lenght 8.
        Each element in this tensor denotes the maximum number of instances present in a specific instance class.
        The position index in this tensor oflength 8, denotes a classid.
        For example, max_inst_per_class[0] = 4, means, cid 11 has 4 instances
        The mapping from position index to actualclassid is defined in self.map_pos_index_to_cid
        self.map_pos_index_to_cid = {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18}
        '''

        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        if self.local_iter % self.debug_img_interval == 0:
            activate_visual_debug = True
        else:
            activate_visual_debug = False

        # Train on source images
        clean_losses = self.get_model().forward_train(img,  img_metas,  gt_semantic_seg,  return_feat=True,
                                                      gt_bboxes=gt_bboxes, gt_labels=gt_labels,
                                                      gt_masks=gt_masks, activate_visual_debug=activate_visual_debug)

        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        if not self.share_src_backward:
            clean_loss.backward(retain_graph=self.enable_fdist)
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                seg_grads = [ p.grad.detach().clone() for p in params if p.grad is not None ]
                grad_mag = calc_grad_magnitude(seg_grads)
                mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.share_src_backward:
                clean_loss = clean_loss + feat_loss
            else:
                feat_loss.backward()
                if self.print_grad_magnitude:
                    params = self.get_model().backbone.parameters()
                    fd_grads = [ p.grad.detach() for p in params if p.grad is not None ]
                    fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                    grad_mag = calc_grad_magnitude(fd_grads)
                    mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Shared source backward
        if self.share_src_backward:
            clean_loss.backward()
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        # getting source images predictions for visualization
        if activate_visual_debug:
            SrcPdSem = self.getPdSem(img)
            SrcPdBox, SrcPdLbl, SrcPdMask = self.getPdBox(dev)

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits, out_roi_head = self.get_ema_model().encode_decode_full(target_img, target_img_metas)

        if self.activate_uda_inst_losses:
            inst_pseduo_weight = self.getInstPseudoWeight(inst_pseduo_weight=self.inst_pseduo_weight)
            with torch.no_grad():
                ema_panoptic_label, pseudo_wght_val = get_ema_panoptic_label(
                                                                                out_roi_head,
                                                                                pseudo_threshold=self.pseudo_threshold,
                                                                                map_pos_index_to_cid=self.map_pos_index_to_cid,
                                                                                pred_shape=target_img.shape[2:], dev=dev,
                                                                                thing_list=self.thing_list,
                                                                                label_divisor=self.label_divisor_target,
                                                                                batch_size=batch_size,
                                                                                src_max_inst_per_class=max_inst_per_class,
                                                                                map_cid_to_pos_index=self.map_cid_to_pos_index,
                                                                                inst_pseduo_weight=inst_pseduo_weight,
                                                                            )

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        if not self.switch_off_self_training:
            pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)
        else:
            pseudo_weight = pseudo_weight * torch.zeros(pseudo_prob.shape, device=dev)  # switch off the self-training on target pixels,
                                                                                        # in this setup, only the loss on the source pixels willbe computed
                                                                                        # since the pseudo_weight tensor has all 0s, the loss on the target pixels will be set to 0
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl, mixed_lbl_pan = [None] * batch_size, [None] * batch_size, [None] * batch_size

        # mix_masks = get_class_masks(gt_semantic_seg) if not self.mix_masks_only_thing else get_class_masks_only_thing(gt_semantic_seg)
        mix_masks = get_class_masks(gt_semantic_seg) if not self.swtich_off_mix_sampling else get_class_masks_dummy(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(strong_parameters, data=torch.stack((img[i], target_img[i])), target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(strong_parameters, target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            if self.activate_uda_inst_losses:
                _, mixed_lbl_pan[i] = strong_transform(strong_parameters, target=torch.stack((gt_panoptic_only_thing_classes[i][0], ema_panoptic_label[i])))
            else:
                mixed_lbl_pan[i] = None

        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        mixed_lbl_pan = torch.cat(mixed_lbl_pan)  if self.activate_uda_inst_losses else None

        # get the box and mask pseudo labels for the mixed image
        if self.activate_uda_inst_losses:
            with torch.no_grad():
                mixed_bboxes, mixed_labels, mixed_masks, box_domain_indicator = get_box_mask_pseduo_labels(
                                                                                                            mixed_lbl_pan, dev, batch_size, self.map_cid_to_pos_index,
                                                                                                            self.label_divisor, self.label_divisor_target
                                                                                                            )
        else:
            mixed_bboxes, mixed_labels, mixed_masks, box_domain_indicator = None, None, None, None


        # if there is no valid pseduo bbox and masks then set the pseudo_wght_val to None
        pseudo_wght_val = None if mixed_bboxes is None else pseudo_wght_val
        # Train on mixed images
        mix_losses = self.get_model().forward_train(mixed_img, img_metas,  mixed_lbl,
                                                    seg_weight=pseudo_weight, return_feat=True,
                                                    gt_bboxes=mixed_bboxes, gt_labels=mixed_labels,
                                                    gt_masks=mixed_masks, box_domain_indicator=box_domain_indicator,
                                                    pseudo_wght_val=pseudo_wght_val,
                                                    activate_visual_debug=activate_visual_debug,
                                                    )
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        # getting mixed images predictions for visualization
        if activate_visual_debug:
            TrgPsBox, TrgPsMask = get_ema_box_mask_preds_for_visual(out_roi_head, batch_size=batch_size, dev=dev, pseudo_threshold=self.pseudo_threshold, crop_size=img.shape[2:])
            MixPdSem = self.getPdSem(img)
            MixPdBox, MixPdLbl, MixPdMask = self.getPdBox(dev)


        if activate_visual_debug:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 6, 6
                fig, axs = plt.subplots( rows,  cols, figsize=(3 * cols, 3 * rows), gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0 }, )
                # Source image and ground-truths
                subplotimg(axs[0][0], vis_img[j], 'SrcImg')
                subplotimg(axs[0][1], gt_semantic_seg[j], 'SrcGTSem', cmap='cityscapes')
                subplotimg(axs[0][2], vis_img[j], 'SrcGTBox', bboxes=gt_bboxes[j], labels=gt_labels[j])
                subplotimg(axs[0][3], convertNDMaskTo2DMask(torch.from_numpy(gt_masks[j].masks).to(dev)), 'SrcGTMask', cmap='gray')
                subplotimg(axs[0][4], gt_panoptic_only_thing_classes[j].squeeze(dim=0), 'SrcGTPan', isPanoptic=True, runner_mode='train', label_divisor=1000)
                # Source image predicitons (Pd)
                subplotimg(axs[1][1], SrcPdSem[j], 'SrcPdSem', cmap='cityscapes')
                subplotimg(axs[1][2], vis_img[j], 'SrcPdBox', bboxes=SrcPdBox[j], labels=SrcPdLbl[j])
                subplotimg(axs[1][3], convertNDMaskTo2DMask(SrcPdMask[j]), 'SrcPdMask', cmap='gray')
                # Target image and teacher network (EMA model) prediction (or pseduo labels)
                subplotimg(axs[2][0], vis_trg_img[j], 'TrgImg')
                subplotimg(axs[2][1], pseudo_label[j], 'TrgPsSem', cmap='cityscapes')
                subplotimg(axs[2][2], vis_trg_img[j], 'TrgPsBox', bboxes=TrgPsBox[j], labels=None)
                subplotimg(axs[2][3], convertNDMaskTo2DMask(TrgPsMask[j]), 'TrgPsMask', cmap='gray')
                if self.activate_uda_inst_losses:
                    subplotimg(axs[2][4], ema_panoptic_label[j], 'TrgPsPan', isPanoptic=True, runner_mode='train', label_divisor=10000) # for target label_divisor=10k and for src, it is 1k
                # Mixed image and pseduo labels (Ps)
                subplotimg(axs[3][0], vis_mixed_img[j], 'MixImg')
                subplotimg(axs[3][1], mixed_lbl[j], 'MixPsSem',  cmap='cityscapes')
                if mixed_bboxes:
                    subplotimg(axs[3][2], vis_mixed_img[j], 'MixPsBox', bboxes=mixed_bboxes[j], labels=mixed_labels[j],
                               box_domain_indicator=box_domain_indicator[j],  pseduo_weight=pseudo_wght_val[j])
                    subplotimg(axs[3][3], convertNDMaskTo2DMask(torch.from_numpy(mixed_masks[j].masks).to(dev)), 'MixPsMask', cmap='gray')
                    subplotimg(axs[3][4], mixed_lbl_pan[j], 'MixPsPan', isPanoptic=True, runner_mode='train', )
                # Mixed image predictions
                subplotimg(axs[4][1], MixPdSem[j], 'MixPdSem', cmap='cityscapes')
                subplotimg(axs[4][2], vis_mixed_img[j], 'MixPdBox', bboxes=MixPdBox[j], labels=MixPdLbl[j])
                subplotimg(axs[4][3], convertNDMaskTo2DMask(MixPdMask[j]), 'MixPdMask', cmap='gray')
                # Others: MixMask, PsWgtSem, FDistMask, ScaledGT
                subplotimg(axs[5][0], mix_masks[j][0], 'MixMask', cmap='gray')
                subplotimg(axs[5][1], pseudo_weight[j], 'PsWgtSem', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(axs[5][2], self.debug_fdist_mask[j][0], 'FDistMask', cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(axs[5][3], self.debug_gt_rescale[j], 'ScaledGT', cmap='cityscapes')
                ###
                for ax in axs.flat:
                    ax.axis('off')
                out_vis_fname = os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png')
                plt.savefig(out_vis_fname)
                plt.close()
        self.local_iter += 1

        return log_vars
