import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.core import save_panoptic_eval, save_bbox_eval

import numpy as np
import cv2
from mmdet.datasets.cityscapes import PALETTE
from PIL import Image
import os
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries

from cityscapesscripts.helpers.labels import labels

import json

PALETTE.append([0,0,0])
colors = np.array(PALETTE, dtype=np.uint8)

def save_segmentation_color_images(result, save_dir=None):
    if save_dir is None:
        root_dir = "tmpDir/tmp_out/"
    else:
        root_dir = os.path.join(save_dir, "tmpDir/tmp_out/")
    os.makedirs(root_dir, exist_ok=True)

    pan_pred, cat_pred, _ = result[0]

    sem = cat_pred[pan_pred].numpy()
    sem_tmp = sem.copy()
    sem_tmp[sem==255] = colors.shape[0] - 1
    sem_img = Image.fromarray(colors[sem_tmp])

    is_background = (sem < 11) | (sem == 255)
    pan_pred = pan_pred.numpy() 
    pan_pred[is_background] = 0

    contours = find_boundaries(pan_pred, mode="outer", background=0).astype(np.uint8) * 255
    contours = dilation(contours)

    contours = np.expand_dims(contours, -1).repeat(4, -1)
    contours_img = Image.fromarray(contours, mode="RGBA")

    out = Image.blend(sem_img, sem_img, 0).convert(mode="RGBA")
    out = Image.alpha_composite(out, contours_img).convert(mode="RGB")
    img_shape = result[0][-1][0]["ori_shape"][0:2][::-1]

    out = cv2.resize(np.array(out)[:,:,::-1], img_shape)
    img_file = result[0][-1][0]["filename"].split("/")[-1]

    output_img_file = img_file.split(".png")[0] + "_out.png"

    cv2.imwrite(os.path.join(root_dir, output_img_file), out)

def output_gt_bbox_for_comparison(save_dir, data_loader):
    gt_folder = os.path.join(save_dir, "groundtruths/")
    if os.path.exists(gt_folder):
        shutil.rmtree(gt_folder)
    full_save_dir = gt_folder
    os.makedirs(full_save_dir, exist_ok=True)
    annotation_file = data_loader.dataset.ann_file

    with open(annotation_file) as f:
        annotation_info = json.load(f)

    annotations = annotation_info["annotations"]
    
    file_id_name_mappings = dict()
    for k in range(len(annotation_info["images"])):
        file_id_name_mappings[annotation_info["images"][k]["id"]] = annotation_info["images"][k]["file_name"]

    # [annotation_info["images"][k]["file_name"] for k in range(len(annotation_info["images"])) if annotation_info["images"][k]["id"] == ann["image_id"]][0]



    annotation_file_mappings = dict()
    for ann in annotations:
        label_id = data_loader.dataset.cat2label[ann["category_id"]]
        bbox = ann["bbox"]
        width = ann["width"]
        height = ann["height"]
        image_file_id = ann["image_id"]
        # file_name = [annotation_info["images"][k]["file_name"] for k in range(len(annotation_info["images"])) if annotation_info["images"][k]["id"] == ann["id"]][0]
        file_name = file_id_name_mappings[ann["image_id"]]
        file_name = file_name.split(".")[0]
        if "/" in file_name:
            file_name = file_name.split("/")[-1]

        if file_name not in annotation_file_mappings:
            annotation_file_mappings[file_name] = []
        
        annotation_file_mappings[file_name].append([label_id, bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]])

    for file_id in annotation_file_mappings:
        curr_annotations = annotation_file_mappings[file_id]
        curr_output_file = os.path.join(full_save_dir, str(file_id) + ".txt")
        with open(curr_output_file, "w") as f:
            for annotation in curr_annotations:
                idx = 0
                line = ""

                for ann in annotation:
                    if idx >= 1:
                        line += " "
                    line += str(ann)
                    idx += 1
                f.write(line + "\n")

            f.close()

def single_gpu_test(model, data_loader, show=False, eval=None, save_dir=None):
    model.eval()
    results = []
    filename_ls = []
    dataset = data_loader.dataset
    label2cat_mappings = dict()
    
    label_mappings = dict()
    for label in labels:
        label_mappings[label.trainId] = label.id
    for cat in dataset.cat2label:
        label2cat_mappings[dataset.cat2label[cat]-1] = label_mappings[cat]
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # print("img_name::", data['img_metas'][0]._data[0][0]["filename"])
        # if not "000079_10" in data['img_metas'][0]._data[0][0]["filename"]:
        #     continue
        # if not data['img_metas'][0].data[0][0]['filename'].endswith("frankfurt_000001_000538.png"):
        #     continue

        with torch.no_grad():
            if eval is not None:
                data['eval'] = eval
            result = model(return_loss=False, rescale=not show, **data)
        if eval is None:
            # file_name = data['img_metas'][0].data[0][0]['filename']
            results.append(result)
            filename_ls.append(data['img_metas'][0].data)
            save_bbox_eval(label2cat_mappings, result, data["img_metas"][0].data[0][0], save_dir)
        else:
            save_panoptic_eval(result, save_dir)
            save_segmentation_color_images(result, save_dir)
        # if show:
        #     model.module.show_result(data, result)
        
        batch_size = data['img'][0].size(0)

        for _ in range(batch_size):
            prog_bar.update()

    # with open(os.path.join(save_dir, "output_res"), "w") as f:
    #     pickle.dump(results, f)

    return results, filename_ls


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, eval=None):
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
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if eval is not None:
                data['eval'] = eval
            result = model(return_loss=False, rescale=True, **data)
        if eval is None:
            results.append(result)
        else:
            save_panoptic_eval(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
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
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
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
