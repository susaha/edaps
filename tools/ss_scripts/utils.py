from mmseg.models.utils.dacs_transforms import denorm, get_mean_std
import torch
import numpy as np
from tools.panoptic_deeplab.save_annotations import label_to_color_image, random_color, flow_compute_color
from tools.panoptic_deeplab.utils import create_label_colormap
from matplotlib import pyplot as plt
import os


def visualize_imgs_labels(data_batch):
    '''
    data_batch['img'] is in the range of 0 to 1 and not in the range of 0 to 255
    so for some functions where we use image as the background and overlay center and offset on it, we need to map the pixel values from
    0-1 range to 0-255 range. More specifally we pass img as images_source_vis[bid]*255, e.g.:
    label_center_source_vis = get_center_visual(images_source_vis[bid]*255, label_center_source[bid][0][0].detach().cpu().numpy().copy(), ratio=0.5, )
    '''
    # SOURCE
    img_metas = data_batch['img_metas']
    dev = data_batch['img'].device
    means, stds = get_mean_std(img_metas, dev)
    img_src = data_batch['img']
    label_source = data_batch['gt_semantic_seg']
    images_source_vis = torch.clamp(denorm(img_src, means, stds), 0, 1).squeeze().cpu().numpy().copy().transpose((0,2,3,1))
    label_instance_source = data_batch['gt_instance_seg']
    label_center_source = data_batch['gt_center']
    center_w_source = data_batch['center_weights']
    label_offset_source = data_batch['gt_offset']
    offset_w_source = data_batch['offset_weights']
    label_depth_source = data_batch['gt_depth_map']
    label_foreground_source = data_batch['gt_foreground_seg']
    # TARGET
    # img_metas_target = data_batch['target_img_metas']
    img_trg = data_batch['target_img']
    label_target = data_batch['target_gt_semantic_seg']
    images_target_vis = torch.clamp(denorm(img_trg, means, stds), 0, 1).squeeze().cpu().numpy().copy().transpose((0, 2, 3, 1))
    label_instance_target = data_batch['target_gt_instance_seg']
    label_center_target = data_batch['target_gt_center']
    center_w_target = data_batch['target_center_weights']
    label_offset_target = data_batch['target_gt_offset']
    offset_w_target = data_batch['target_offset_weights']
    label_foreground_target = data_batch['target_gt_foreground_seg']
    #
    rows, cols = 6, 6
    out_dir = '/home/suman/Downloads/work_march_27_2022'
    for bid in range(2):
        fnames = img_metas[bid]['ori_filename'].split('.')[0]
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0.0, 'right': 1, 'left': 0}, )
        # SOURCE
        label_source_vis = get_semantic_visual(label_source[bid].squeeze().detach().cpu().numpy().copy(), colormap=create_label_colormap())
        label_instance_source_vis = get_instance_visual(label_instance_source[bid].squeeze().detach().cpu().numpy().copy().astype(int), )
        label_center_source_vis = get_center_visual(images_source_vis[bid]*255, label_center_source[bid][0][0].detach().cpu().numpy().copy(), ratio=0.5, )
        center_w_source_vis = get_heatmap_visual(images_source_vis[bid]*255, center_w_source[bid][0].detach().cpu().numpy().copy(), ratio=0.5, )
        label_offset_source_vis = get_offset_visual(images_source_vis[bid]*255, label_offset_source[bid][0].detach().cpu().numpy().copy().transpose(1, 2, 0), ratio=0.5, )
        offset_w_source_vis = get_heatmap_visual(images_source_vis[bid] * 255, offset_w_source[bid][0].detach().cpu().numpy().copy(), ratio=0.5, )
        label_depth_source_vis = (colorize(label_depth_source[bid][0].detach().cpu().numpy().copy(), cmap="plasma") * 255).astype(np.uint8)
        label_foreground_source_vis = get_heatmap_visual(images_source_vis[bid] * 255, label_foreground_source[bid][0].detach().cpu().numpy().copy(), ratio=0.5, )
        # TARGET
        label_target_vis = get_semantic_visual(label_target[bid].squeeze().detach().cpu().numpy().copy(), colormap=create_label_colormap())
        label_instance_target_vis = get_instance_visual(label_instance_target[bid].squeeze().detach().cpu().numpy().copy().astype(int), )
        label_center_target_vis = get_center_visual(images_target_vis[bid] * 255, label_center_target[bid][0][0].detach().cpu().numpy().copy(), ratio=0.5, )
        center_w_target_vis = get_heatmap_visual(images_target_vis[bid] * 255, center_w_target[bid][0].detach().cpu().numpy().copy(), ratio=0.5, )
        label_offset_target_vis = get_offset_visual(images_target_vis[bid] * 255, label_offset_target[bid][0].detach().cpu().numpy().copy().transpose(1, 2, 0), ratio=0.5, )
        offset_w_target_vis = get_heatmap_visual(images_target_vis[bid] * 255, offset_w_target[bid][0].detach().cpu().numpy().copy(), ratio=0.5, )
        label_foreground_target_vis = get_heatmap_visual(images_target_vis[bid] * 255, label_foreground_target[bid][0].detach().cpu().numpy().copy(), ratio=0.5, )
        # Source image and labels
        subplotimg(axs[0][0], images_source_vis[bid], 'Src_Img')
        subplotimg(axs[0][1], label_source_vis, 'Src_Sem_Lbl')
        subplotimg(axs[0][2], label_instance_source_vis, 'Src_Ins_Lbl')
        subplotimg(axs[0][3], label_center_source_vis, 'Src_Cnt_Lbl')
        subplotimg(axs[0][4], label_offset_source_vis, 'Src_Ofs_Lbl')
        subplotimg(axs[0][5], label_depth_source_vis, 'Src_Dep_Lbl')
        # Source center and offset weights, gt_foreground_seg
        subplotimg(axs[1][2], label_foreground_source_vis, 'Src_Foreground')
        subplotimg(axs[1][3], center_w_source_vis, 'Src_Cnt_W')
        subplotimg(axs[1][4], offset_w_source_vis, 'Src_Ofs_W')
        # target image and labels
        subplotimg(axs[2][0], images_target_vis[bid], 'Src_Img')
        subplotimg(axs[2][1], label_target_vis, 'Src_Sem_Lbl')
        subplotimg(axs[2][2], label_instance_target_vis, 'Src_Ins_Lbl')
        subplotimg(axs[2][3], label_center_target_vis, 'Src_Cnt_Lbl')
        subplotimg(axs[2][4], label_offset_target_vis, 'Src_Ofs_Lbl')
        # subplotimg(axs[2][5], label_depth_target_vis, 'Src_Dep_Lbl')
        # target center and offset weights, gt_foreground_seg
        subplotimg(axs[3][2], label_foreground_target_vis, 'Src_Foreground')
        subplotimg(axs[3][3], center_w_target_vis, 'Src_Cnt_W')
        subplotimg(axs[3][4], offset_w_target_vis, 'Src_Ofs_W')
        #
        for ax in axs.flat:
            ax.axis('off')
        out_fname = os.path.join(out_dir, f'{fnames}_bid_{bid}.png')
        plt.savefig(out_fname)
        plt.close()
    # print()


def subplotimg(ax, img, title, palette=None, **kwargs):
    ax.imshow(img, **kwargs)
    ax.set_title(title)

def get_semantic_visual(label, add_colormap=True, normalize_to_unit_values=False, scale_values=False, colormap=None, image=None, blend_ratio=0.5, ):
    # Add colormap for visualizing the prediction.
    if add_colormap:
        colored_label = label_to_color_image(label, colormap)
    else:
        colored_label = label
    if normalize_to_unit_values: # False
        min_value = np.amin(colored_label)
        max_value = np.amax(colored_label)
        range_value = max_value - min_value
        if range_value != 0:
            colored_label = (colored_label - min_value) / range_value
    if scale_values: # False
        colored_label = 255. * colored_label
    if image is not None:
        colored_label = blend_ratio * colored_label + (1 - blend_ratio) * image
    return colored_label.astype(dtype=np.uint8)


def get_instance_visual(label,stuff_id=0,image=None, blend_ratio=0.5,):
    # Add colormap for visualizing the prediction.
    ids = np.unique(label)
    num_colors = len(ids)
    colormap = np.zeros((num_colors, 3), dtype=np.uint8)
    # Maps label to continuous value.
    for i in range(num_colors):
        label[label == ids[i]] = i
        colormap[i, :] = random_color(rgb=True, maximum=255)
        if ids[i] == stuff_id:
            colormap[i, :] =  np.array([0, 0, 0]) # np.array([255, 0, 0])
    colored_label = colormap[label]
    if image is not None:
        colored_label = blend_ratio * colored_label + (1 - blend_ratio) * image
    return colored_label.astype(dtype=np.uint8)


def get_center_visual(image, center_heatmap, ratio=0.5,):
    center_heatmap = center_heatmap[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
    center_heatmap = center_heatmap.clip(0, 255)
    image = ratio * center_heatmap + (1 - ratio) * image
    return image.astype(dtype=np.uint8)


def get_offset_visual(image, offset, ratio=0.5,):
    offset_image = flow_compute_color(offset[:, :, 1], offset[:, :, 0])
    image = ratio * offset_image + (1 - ratio) * image
    return image.astype(dtype=np.uint8)


def get_heatmap_visual(image, center_heatmap, ratio=0.5,):
    center_heatmap = center_heatmap[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
    center_heatmap = center_heatmap.clip(0, 255)
    image = ratio * center_heatmap + (1 - ratio) * image
    return image.astype(dtype=np.uint8)


def get_domain_mask_visual(center_heatmap):
    center_heatmap = center_heatmap[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
    center_heatmap = center_heatmap.clip(0, 255)
    image = center_heatmap
    return image.astype(dtype=np.uint8)

def colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image