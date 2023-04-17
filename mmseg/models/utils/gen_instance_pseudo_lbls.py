import torch
import torch.nn.functional as F
import numpy as np

from mmseg.models.utils.dacs_transforms import strong_transform

'''
NOTE-1: 
    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    # torch.argmin(distance, dim=0) returns a tensor of shape [H*W]
    # it returns the center id (between 0 to K-1) for the center which has the minimum distance from a pixel in ctr_loc
    # e.g. if there are 17 centers i.e. K=17, then distance is a tensor of shape H*W each element in this center has a value between 0 and 16 (i.e. 17-1)
    # now if you reshape this tensor H*W to [1,H,W] then you get the K segments, each segment has group of pixels with segment id starting from 0 to 16
    # you can offset the segment id by addining a offset value, e.g. 1

    # instance_id_offset is used to assign the instance segment id from a number starting from instance_id_offset
    # this is required to have unqiue instance ids for each instance segments in the cross-domain mixed image (CDMI)
    # the source instance id starts from 1,2,.., N, and the instance_id_offset = N+1
    # i.e. if there are 10 instances in the source part of the CDMI, then the instance_id_offset=11,
    # which will start the instance segmnet id for the target part starting from 11,12,... and so on
'''


class CenterAndOffsetTargetGenerator(object):
    def __init__(self, sigma=8, device=None):
        # self.logger = logger
        # self.logger.info('ctrl/utils_center_offset_pseudo_labels.py --> class CenterAndOffsetTargetGenerator() : __init__()')
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.g = torch.from_numpy(self.g).to(device)

    def __call__(self, instance_seg_map_batch):
        instance_seg_map_batch = instance_seg_map_batch.squeeze(dim=1)
        # TODO: CHNAGE FROM NUMPY CODE TO PYTORCH CODE
        batch_size, height, width = instance_seg_map_batch.shape[0], instance_seg_map_batch.shape[1], instance_seg_map_batch.shape[2]
        offset = torch.zeros(batch_size, 2, height, width).to(instance_seg_map_batch.device)
        center = torch.zeros(batch_size, 1, height, width).to(instance_seg_map_batch.device)
        # generates a coordinate map, where each location is the coordinate of that loc
        y_coord = torch.arange(height, dtype=instance_seg_map_batch.dtype, device=instance_seg_map_batch.device).repeat(1, width, 1).transpose(1, 2)
        x_coord = torch.arange(width, dtype=instance_seg_map_batch.dtype, device=instance_seg_map_batch.device).repeat(1, height, 1)
        x_coord = x_coord.squeeze(dim=0)
        y_coord = y_coord.squeeze(dim=0)
        for bid in range(batch_size):
            instance_seg_map = instance_seg_map_batch[bid]
            inst_ids = torch.unique(instance_seg_map)
            for id in inst_ids:
                if id == 0:
                    continue
                mask_index = torch.where(instance_seg_map == id)  # returns tuple of LongTensor
                assert len(mask_index[0]) != 0, 'there should be at least one pixel in one segment'
                center_y, center_x = torch.mean(mask_index[0].float()), torch.mean(mask_index[1].float())
                # generate ground truth center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left corner point: ul[0] -> y coord or height, ul[1] -> x coord or width
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right corner point: br[0] -> y coord or height, br[1] -> x coord or width
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
                # get the indices
                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                # generate 51x51 gaussian window around each center (y,x)
                center[bid][0, aa:bb, cc:dd] = torch.maximum(center[bid][0, aa:bb, cc:dd], self.g[a:b, c:d])
                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (torch.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (torch.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[bid][offset_y_index] = center_y - y_coord[mask_index]
                offset[bid][offset_x_index] = center_x - x_coord[mask_index]

        return center, offset


def group_pixels(ctr_batch, offsets_batch, coord, target_instance_id_offsets=None):
    '''
    note that if a segment has instance id 0, that means it is a VOID segment or invaid segment
    a valid instance segment  must have a minimum instance segment id of 1
    target_instance_id_offsets is generated in such a way that if there are 0 instances in source image then
    target_instance_id_offsets is set 1, if there are N instances in source image, then target_instance_id_offsets
    is set to N+1, so 0 denoted a VOID segment
    '''
    if not target_instance_id_offsets:
        raise ValueError('provide valid value for target_instance_id_offsets!')
    height, width = offsets_batch.size()[2:]  # offsets_batch:tensor:(2,2,H,W)
    batch_size = offsets_batch.size()[0]
    instance_id = torch.zeros((batch_size, height, width)).to(offsets_batch.device)
    for bid in range(batch_size):
        ctr = ctr_batch[bid]  # ctr: [K, 2]
        # if there is no center for this image then continue
        if ctr.size(0) == 0:
            continue
        ctr_loc = coord + offsets_batch[bid, :]  # coord:tensor:(2,H,W) , offsets:tensor:(2,H,W), ctr_loc:tensor:(2,H,W)
        ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)  # ctr_loc:tensor: (H*W, 2)
        ctr = ctr.unsqueeze(1)  # ctr: [K, 2] -> [K, 1, 2], K is the totalno.of centers
        ctr_loc = ctr_loc.unsqueeze(0)  # ctr_loc = [H*W, 2] -> [1, H*W, 2]
        # distance: [K, H*W], K is the no. of centers, e.g. [17, 1024*2048]
        distance = torch.norm(ctr - ctr_loc, dim=-1)
        # generating the instance segmentation map - for more details refer to the above NOTE-1
        instance_id[bid, :] = torch.argmin(distance, dim=0).reshape((height, width)) + target_instance_id_offsets[bid]
    # instance segmentation map where each instance segment is assigned with
    # an unique instance id starting from target_instance_id_offsets,
    # target_instance_id_offsets+1, target_instance_id_offsets+2. ..., M
    return instance_id


def find_instance_center(ctr_hmp_batch, threshold=0.1, nms_kernel=3, top_k=None):
    batch_size = ctr_hmp_batch.size(0)  # ctr_hmp_batch: tensor: 2x1xHxW
    ctr_all_batch = {}
    for bid in range(batch_size):
        ctr_hmp = ctr_hmp_batch[bid]
        if ctr_hmp.size(0) != 1:  # False: ctr_hmp: tensor(1,1,1024,2048)
            raise ValueError('Only supports inference for batch size = 1')
        # thresholding, setting values below threshold to -1
        ctr_hmp = F.threshold(ctr_hmp, threshold, -1)  # ctr_hmp: tensor(1,1,1024,2048)
        # NMS
        nms_padding = int((nms_kernel - 1) // 2)  # nms_padding:3
        ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)  # ctr_hmp_max_pooled: tensor(1,1,1024,2048)
        ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1  # ctr_hmp: tensor(1,1,1024,2048)
        # squeeze first two dimensions
        ctr_hmp = ctr_hmp.squeeze()  # ctr_hmp: tensor(1024,2048) # this is only applicable for single batch, we need to do it for batch size 2
        assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'
        # find non-zero elements
        ctr_all = torch.nonzero(ctr_hmp > 0)  # ctr_all: tensor (N,2), N is the no of center > 0
        if top_k is None:  # False
            ctr_all_batch[bid] = ctr_all
        elif ctr_all.size(0) < top_k:  # True, e.g. ctr_all: tensor(17,2), and top_k = 200,
            ctr_all_batch[bid] = ctr_all
        else:
            # find top k centers.
            # ctr_hmp: tensor(1024,2048), top_k=200, torch.flatten(ctr_hmp) gives you a tensor of shape 1024x2048, top_k_scores: tensor(k,)
            # having scores in descending order, the top scoreis the first element e.g. 0.94, 0.92, 0.89
            top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
            ctr_all_batch[bid] = torch.nonzero(ctr_hmp > top_k_scores[-1])
    return ctr_all_batch


def get_instance_segmentation(semantic_prediction, ctr_hmp, offsets, thing_list, coord, threshold=0.1, nms_kernel=3, top_k=None, target_instance_id_offsets=None):
    # gets foreground segmentation
    thing_seg = torch.zeros_like(semantic_prediction)  # semantic_prediction: tensor:(1,1024,2048)
    for thing_class in thing_list:
        thing_seg[semantic_prediction == thing_class] = 1  # finding the thing mask from the semantic prediction : thing_seg: (1,1024,2048)

    # get the centers from predicted center heatmap :ctr_hmp
    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)  # ctr_batch[0]=tensor:(K,2), ctr_batch[1]=tensor:(K,2), K is the no of center

    NO_VALID_SEGMENT = False
    batch_size = len(ctr)
    for bid in range(batch_size):
        if ctr[bid].size(0) == 0:
            NO_VALID_SEGMENT = True
            break

    # if there is no valid segment in one of the two images,
    # we dont train the network with unlabeled center and offset losses
    # initially for certain no of iterations the unlabled center and offset losses are
    # not computed as we don't get any valid predictions for the center
    # sometimes we get center predictions but they are not valid since they are
    # on stuff regions,which are then discarded by the semanitc predictions on the target
    # if the semantic predictions on target images are wrong then we compute wrong center and offset losses
    # which might be the case initially , but we hope that as our unlabled semantic loss goes down
    # it learns better semantics on target and thus allows to learn better center and offset
    if NO_VALID_SEGMENT:
        # if the teacher networks does not predicts valid centers
        # then return the dummy instance segmentation map, center and offset weights
        batch_size, height, width = semantic_prediction.shape
        center_weights_dummy = torch.zeros(batch_size, height, width).to(semantic_prediction.device)
        offset_weights_dummy = torch.zeros(batch_size, height, width).to(semantic_prediction.device)
        instance_prediction_dummy = torch.zeros_like(semantic_prediction)
        return instance_prediction_dummy, center_weights_dummy, offset_weights_dummy, NO_VALID_SEGMENT

    # once centers are extracted,then generate the final instance segmentation map using the predicted offsets
    ins_seg = group_pixels(ctr, offsets, coord, target_instance_id_offsets=target_instance_id_offsets)

    # final predicted instance segmentation map
    instance_prediction = thing_seg * ins_seg

    '''
    the instance segmentation map "ins_seg" returned by the group_pixels() function might have segment ids 0,1
    but after correcting the "ins_seg" by multiplying it with thing_seg, it might now have no valid segment
    the reason being it might have some instance segment with instance id 1 but the semantic prediction tells us that
    the region belongs to stuff class, in that case, after multiplying  "ins_seg" with  "thing_seg" the pixels present in
    that segment are set to 0 because thing_seg has onlyvalue 1 where there are pixels predicted as thing class
    and for pixel which are predicted as stuff class, are set to all 0s.
    So, the main point here, even if you have some valid center points returned by the find_instance_center() function
    after the multiplication (i.e. instance_prediction = thing_seg * ins_seg), you might end up with no valid predicted segment in the
    image, you need to capture that
    '''

    # generating the center and offset weights
    batch_size, height, width = instance_prediction.shape
    center_weights = torch.ones(batch_size, height, width).to(instance_prediction.device)
    offset_weights = torch.zeros(batch_size, height, width).to(instance_prediction.device)
    # NO_VALID_SEGMENT = False
    VALID_SEGMENT = True
    for bid in range(batch_size):
        uids = instance_prediction[bid].unique()
        # if there is only one segment with instance id 0, 0 denotes invalid or VOID instance id
        # that means this image does not have any valid instance segments
        if len(uids) == 1 and uids[0] == 0:
            # NO_VALID_SEGMENT = True
            VALID_SEGMENT = False
            break
        for uid in uids:
            if uid == 0:
                continue
            offset_weights[bid][instance_prediction[bid] == uid] = 1

    # if NO_VALID_SEGMENT:
    #     center_weights[:] = 0
    #     offset_weights[:] = 0
    #     instance_prediction[:] = 0
    if not VALID_SEGMENT:
        center_weights[:] = 0
        offset_weights[:] = 0
        instance_prediction[:] = 0

    '''
    instance_prediction: bs x h x w
    center_weights:  bs x h x w
    offset_weights:  bs x h x w
    '''

    return instance_prediction, center_weights, offset_weights, VALID_SEGMENT



def get_mixed_lbls(batch_size, gt_instance_seg, pseudo_label, ema_center_logits, ema_offset_logits,
                   mix_masks, center_offset_target_generator, center_weight_src, offset_weight_src,
                   strong_parameters, pseudo_weight_cnt, pseudo_weight_ofs, center_threshold):

    crop_height = 512
    crop_width = 512
    DEVICE = gt_instance_seg.device
    y_coord = torch.arange(crop_height, dtype=torch.float32, device=DEVICE).repeat(1, crop_width, 1).transpose(1, 2)
    x_coord = torch.arange(crop_width, dtype=torch.float32, device=DEVICE).repeat(1, crop_height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)
    #
    cityscapes_thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
    CENTER_THRESHOLD = center_threshold
    NMS_KERNEL = 7
    TOP_K_INSTANCE = 200
    #
    gt_instance_seg = gt_instance_seg.squeeze(dim=1)  # [B,1,512,512] --> [B,512,512]
    target_instance_id_offsets = []
    for bid in range(batch_size):
        target_instance_id_offsets.append(torch.unique(gt_instance_seg[bid, :])[-1] + 1)

    # get the instance segmentation map for the target image
    with torch.no_grad():
        inst_pred_target, center_weight_trg, \
        offset_weight_trg, VALID_SEGMENT = get_instance_segmentation(
            pseudo_label,
            ema_center_logits,
            ema_offset_logits,
            cityscapes_thing_list,
            coord,
            threshold=CENTER_THRESHOLD,
            nms_kernel=NMS_KERNEL,
            top_k=TOP_K_INSTANCE,
            target_instance_id_offsets=target_instance_id_offsets,
        )

    # now here generate the pseudo instance seg label (mixed_lbl_inst) for the mixed image
    mixed_lbl_inst, mixed_lbl_cnt_w, mixed_lbl_ofs_w, mixed_lbl_depth, mixed_lbl_cnt, mixed_lbl_ofs \
        = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size

    if VALID_SEGMENT:
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            _, mixed_lbl_inst[i] = strong_transform(strong_parameters, target=torch.stack((gt_instance_seg[i], inst_pred_target[i])))
            _, mixed_lbl_cnt_w[i] = strong_transform(strong_parameters, target=torch.stack((center_weight_src[i][0], center_weight_trg[i] * pseudo_weight_cnt[i])))
            _, mixed_lbl_ofs_w[i] = strong_transform(strong_parameters, target=torch.stack((offset_weight_src[i][0], offset_weight_trg[i] * pseudo_weight_ofs[i])))
        mixed_lbl_inst = torch.cat(mixed_lbl_inst)
        mixed_lbl_cnt_w = torch.cat(mixed_lbl_cnt_w)
        mixed_lbl_ofs_w = torch.cat(mixed_lbl_ofs_w)
        #
        with torch.no_grad():
            mixed_lbl_cnt, mixed_lbl_ofs = center_offset_target_generator(mixed_lbl_inst.detach())

    return mixed_lbl_cnt, mixed_lbl_cnt_w, mixed_lbl_ofs, mixed_lbl_ofs_w, mixed_lbl_inst, mixed_lbl_depth, VALID_SEGMENT
