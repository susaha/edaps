# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted from OpenMMLab: https://github.com/open-mmlab
# Modifications: Support for panoptic segmentation
# ------------------------------------------------------------------------------------

import os.path as osp
import pickle
import shutil
import tempfile
import time
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
from tools.panoptic_deeplab.post_processing import get_cityscapes_instance_format_for_maskrcnn_v3
from tools.panoptic_deeplab.post_processing import get_cityscapes_instance_format_for_maskrcnn
import os
import PIL.Image as Image
import numpy as np
from tools.panoptic_deeplab.save_annotations import random_color
from matplotlib import pyplot as plt
from mmseg.utils.visualize_pred  import subplotimg
from tools.panoptic_deeplab.post_processing import merge_semantic_and_instance, merge_semantic_and_instance_v2
from mmseg.utils.visualize_pred import prep_pan_for_vis
import cv2


def single_gpu_test_uda_dump_results_to_disk(model,
                    data_loader,
                    out_dir=None,
                    debug=False,
                    dataset_name=None,
                    logger=None,
                    ):
    model.eval()
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if debug or dataset_name == 'mapillary':
                rescale = False
            else:
                rescale = True
            result = model(return_loss=False, rescale=rescale, **data)
            img_file_path = data['img_metas'][0].data[0][0]['filename']
            img_filename = img_file_path.split('/')[-1]
            npy_fname = os.path.join(out_dir, img_filename.replace('.jpg', '.npy'))
            np.save(npy_fname, result)
            logger.info(f'predictions saved at : {npy_fname}')
        # batch_size = len(result)
        # for _ in range(batch_size):
        #     prog_bar.update()
    return True

def single_gpu_test_uda_for_visual_debug(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    debug=False,
                    show_score_thr=0.3,
                    dataset_name=None,
                    panop_eval_temp_folder=None,
                    ):


    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if True:
            with torch.no_grad():
                if debug or dataset_name == 'mapillary':
                    rescale = False
                else:
                    rescale = True
                # Forward call for inference
                results = model(return_loss=False, rescale=rescale, **data)
            i = 0
            mask_th = 0.95
            stuff_id = 0
            train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0]
            thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            use_semantic_decoder_for_instance_labeling = False
            use_semantic_decoder_for_panoptic_labeling = False
            image_filename_list = []
            image_filename_list.append(data['img_metas'][0].data[0][0]['ori_filename'])
            if dataset_name == 'cityscapes':
                image_filename = image_filename_list[i].split('/')[1]
                image_filename = image_filename.split('.')[0]
            elif dataset_name == 'mapillary':
                image_filename = image_filename_list[i]
            out_dict = {}
            # creating folders to dump PNGs generated during panoptc-deeplab evaluation
            eval_folder = {}
            eval_folder['instance'] = os.path.join(panop_eval_temp_folder, 'instance')
            eval_folder['visuals'] = os.path.join(panop_eval_temp_folder, 'visuals')
            eval_folder['semantic'] = os.path.join(panop_eval_temp_folder, 'semantic')
            eval_folder['panoptic'] = os.path.join(panop_eval_temp_folder, 'panoptic')
            # instance seg
            out_dict['boxes'] = results[i]['ins_results'][0][0]
            out_dict['masks'] = results[i]['ins_results'][0][1]
            # semantic seg
            pred_shape = results[i]['sem_results'][0].shape
            out_dict['semantic'] = results[i]['sem_results'][0]
            mask_th=0.95
            instances, ins_seg, pan_seg_thing_classes = \
                get_cityscapes_instance_format_for_maskrcnn(
                    out_dict['boxes'],
                    out_dict['masks'],
                    pred_shape=pred_shape,
                    mask_score_th=mask_th,
                    sem_seg=out_dict['semantic'],
                    device=device,
                    thing_list=thing_list,
                    use_semantic_decoder_for_instance_labeling=use_semantic_decoder_for_instance_labeling,
                    use_semantic_decoder_for_panoptic_labeling=use_semantic_decoder_for_panoptic_labeling,
                    nms_th=None,
                    intersec_th=None,
                )
            # generatig the panoptic segmentation from semantic and instance segs
            out_dict['semantic'] = torch.from_numpy(out_dict['semantic']).long().to(device)
            ins_seg = torch.from_numpy(ins_seg).long().to(device)
            label_divisor = 1000
            stuff_area = 2048
            ignore_label = 255
            panoptic_pred = merge_semantic_and_instance(
                                                        out_dict['semantic'].unsqueeze(dim=0),
                                                        ins_seg.unsqueeze(dim=0),
                                                        label_divisor,
                                                        thing_list,
                                                        stuff_area,
                                                        void_label=label_divisor * ignore_label
                                                    )
            fig, ax = plt.subplots()
            DPI = fig.get_dpi()
            fig.set_size_inches(1024.0 / float(DPI), 512.0 / float(DPI))
            panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()
            panoptic_pred = prep_pan_for_vis(panoptic_pred, dataset_name=dataset_name, debug=debug,
                             blend_ratio=1.0, img=None, runner_mode='val',
                             ax=ax, label_divisor=1000)
            ax.imshow(panoptic_pred)
            cityname = image_filename_list[i].split('/')[0]
            out_dir = os.path.join(eval_folder['visuals'], 'edaps_pred_visuals', cityname)
            os.makedirs(out_dir, exist_ok=True)
            out_vis_fname_pan_Seg = os.path.join(out_dir, f'{image_filename}_panoptic_seg.png')
            fig.savefig(out_vis_fname_pan_Seg)  # save the figure to file
            plt.close(fig)
            print(f'*** file saved at :{out_vis_fname_pan_Seg}')
    return None


def single_gpu_test_uda(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    debug=False,
                    show_score_thr=0.3,
                    dataset_name=None,
                    ):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if debug or dataset_name == 'mapillary':
                rescale = False
            else:
                rescale = True
            result = model(return_loss=False, rescale=rescale, **data)
        batch_size = len(result)
        if False:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        # if show or out_dir:
        if False:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)
        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results)) for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results, encode_mask_results(mask_results))

        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_inference(model, data_loader, tmpdir=None, gpu_collect=False):
    import numpy as np
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.    log_interval = 10
    benchmark_dict = dict(unit='img / s')
    # benchmark_dict = dict(config=args.config, unit='img / s')
    repeat_times = 1
    overall_fps_list = []
    for time_index in range(repeat_times):
        num_warmup = 5
        pure_inf_time = 0
        total_iters = 200
        for i, data in enumerate(data_loader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(
                        f'Done image [{i + 1:<3}/ {total_iters}], '
                        f'fps: {fps:.2f} img / s, '
                        f'mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
            if (i + 1) == total_iters:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.2f} img / s\n')
                benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                overall_fps_list.append(fps)
                break
    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
    print(f'Average fps of {repeat_times} evaluations: '
          f'{benchmark_dict["average_fps"]}')
    print(f'The variance of {repeat_times} evaluations: '
          f'{benchmark_dict["fps_variance"]}')
    # print(benchmark_dict)
    # mmcv.dump(benchmark_dict, json_file, indent=4)

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))
        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)
    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
