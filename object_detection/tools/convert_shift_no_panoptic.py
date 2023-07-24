import argparse
import glob
import json
import shutil
from multiprocessing import Pool, Value, Lock
from os import path, mkdir, listdir
import sys 
import cv2

import numpy as np
import tqdm
from PIL import Image
from cityscapesscripts.helpers.labels import labels as cs_labels
from pycococreatortools import pycococreatortools as pct

parser = argparse.ArgumentParser(description="Convert KITTI to coco format")
parser.add_argument("root_dir", metavar="ROOT_DIR", type=str, help="Root directory of Cityscapes")
parser.add_argument("out_dir", metavar="OUT_DIR", type=str, help="Output directory")

_SPLITS = {
    "train": ("training/images", "training/annotations"),
    "val": ("validation/images", "validation/annotations"),
}
_INSTANCE_EXT = ".jpg"
_IMAGE_EXT = ".jpg"
_EVAL = "shift_panoptic_val"

label_mappings = dict()

def build_label_mappings():
    
    for label in cs_labels:
        label_mappings[label.id] = label.trainId


def main(args):
    print("Loading shift from", args.root_dir)
    global _EVAL
    num_stuff, num_thing = _get_meta()
    build_label_mappings()
    _ensure_dir(args.out_dir)
    ann_dir = path.join(args.out_dir, "annotations")
    _ensure_dir(ann_dir)
    stuff_dir = path.join(args.out_dir, "stuffthingmaps")
    _ensure_dir(stuff_dir)
    _EVAL = path.join(args.out_dir, _EVAL)
    _ensure_dir(_EVAL)


    # COCO-style category list
    coco_categories = []
    categories = []
    for lbl in cs_labels:
        if lbl.ignoreInEval:
            continue
        categories.append({'id': int(lbl.id),
                           'name': lbl.name,
                           'color': lbl.color,
                           'supercategory': lbl.category,
                           'isthing': 1 if lbl.hasInstances else 0})

        if lbl.trainId != 255 and lbl.trainId != -1 and lbl.hasInstances:
            coco_categories.append({
                "id": lbl.trainId,
                "name": lbl.name
            })

    # Process splits
    images = []
    for split, (split_img_subdir, split_msk_subdir) in _SPLITS.items():
        print("Converting", split, "...")

        img_base_dir = path.join(args.root_dir, split_img_subdir)
        annotation_file = path.join(args.root_dir, split_msk_subdir + ".json")
        # img_list = _get_images(img_base_dir)

        img_dir = path.join(args.out_dir, split)
        _ensure_dir(img_dir)

        split_dir = path.join(stuff_dir, split)
        _ensure_dir(split_dir)


        # Convert to COCO detection format
        coco_out = {
            "info": {"version": "1.0"},
            "images": [],
            "categories": coco_categories,
            "annotations": []
        }
        
        images = []
        annotations = []
        # Process images in parallel
        worker = _Worker(img_base_dir, annotation_file, img_dir, split_dir)
        with Pool(initializer=_init_counter, initargs=(_Counter(0),), processes=1) as pool:
            total = len(worker.img_list)
            # for coco_img, coco_ann, city_img, city_ann in tqdm.tqdm(pool.imap(worker, img_list, 8), total=total):

            for img_item in tqdm.tqdm(worker.img_list):
                coco_img, coco_ann, city_img, city_ann = worker(img_item)

            # for coco_img, coco_ann, city_img, city_ann in tqdm.tqdm(worker(img_list), total=total):

                # COCO annotation
                coco_out["images"].append(coco_img)
                coco_out["annotations"] += coco_ann
                images.append(city_img)
                annotations.append(city_ann)

        if split == 'val':
            d = {'images': images,
                 'annotations': annotations,
                 'categories': categories}
      
            with open(_EVAL+'.json', 'w') as f:
                json.dump(d, f, sort_keys=True, indent=4)

        # Write COCO detection format annotation
        with open(path.join(ann_dir, split + ".json"), "w") as fid:
            json.dump(coco_out, fid)


def _get_images(base_dir):
    img_list = []
    for img in glob.glob(path.join(base_dir, "*" + _INSTANCE_EXT)):
        _, img = path.split(img)

        parts = img.split(".")
        img_id = parts[0]
        lbl_cat = 'dummy'
        img_list.append(('dummy', img_id, lbl_cat))

    return img_list


def _get_meta():
    num_stuff = sum(1 for lbl in cs_labels if 0 <= lbl.trainId < 255 and not lbl.hasInstances)
    num_thing = sum(1 for lbl in cs_labels if 0 <= lbl.trainId < 255 and lbl.hasInstances)
    return num_stuff, num_thing


def _ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


class _Worker:
    def __init__(self, img_base_dir, annotation_file, img_dir, msk_dir):
        self.img_base_dir = img_base_dir
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        
        with open(self.annotation_file) as f:
            self.annotation_json = json.load(f)
        
        self.width = self.annotation_json["images"][0]["width"]
        self.height = self.annotation_json["images"][0]["height"]
        self.counter = _Counter(0)
        self.img_list = [self.annotation_json["images"][idx]['id'] for idx in range(len(self.annotation_json["images"]))]

    def __call__(self, img_desc):
        img_id = img_desc
        counter = self.counter
        img_unique_id = counter.increment()
        coco_ann = []
        # Load the annotation
        with Image.open(path.join(self.img_base_dir, img_id + _INSTANCE_EXT)) as lbl_img:
            # lbl = cv2.resize(np.array(lbl_img), (1280, 384), interpolation=cv2.INTER_NEAREST)
            lbl = lbl_img
            lbl_size = np.array(lbl_img).shape[:2][::-1]

        curr_annotations = [self.annotation_json["annotations"][idx] for idx in range(len(self.annotation_json["annotations"])) if self.annotation_json["annotations"][idx]["image_id"] == img_id][0]

        # color_ids = np.vstack({tuple(r) for r in lbl.reshape(-1,3)})

        # Compress the labels and compute cat
        # lbl_out = np.ones(lbl.shape[:2], np.uint8)*255
        cat = [255]
        iscrowd = [0]
        segmInfo = []
        # for color_id in color_ids:
        #     city_id = color_id[2]*256*256 + color_id[1]*256 + color_id[0]
        #     if city_id < 1000:
        #         # Stuff or group
        #         cls_i = city_id
        #         iscrowd_i = cs_labels[cls_i].hasInstances
        #     else:
        #         # Instance
        #         cls_i = city_id // 1000
        #         iscrowd_i = False

        #     # If it's a void class just skip it
        #     if cs_labels[cls_i].trainId == 255 or cs_labels[cls_i].trainId == -1:
        #         continue
        #     if cs_labels[cls_i].trainId == 3:
        #         print (self.img_dir) 
        #     # Extract all necessary information
        #     iss_class_id = cs_labels[cls_i].trainId
            
        #     mask_i = np.logical_and(np.logical_and(lbl[:,:,0]==color_id[0],lbl[:,:,1]==color_id[1]), lbl[:,:,2]==color_id[2])

        #     area = np.sum(mask_i) # segment area computation

        #     # bbox computation for a segment
        #     hor = np.sum(mask_i, axis=0)
        #     hor_idx = np.nonzero(hor)[0]
        #     x = hor_idx[0]
        #     width = hor_idx[-1] - x + 1
        #     vert = np.sum(mask_i, axis=1)
        #     vert_idx = np.nonzero(vert)[0]
        #     y = vert_idx[0]
        #     height = vert_idx[-1] - y + 1
        #     bbox = [int(x), int(y), int(width), int(height)]

        #     segmInfo.append({"id": int(city_id),
        #                      "category_id": int(cls_i),
        #                      "area": int(area),
        #                      "bbox": bbox,
        #                      "iscrowd": iscrowd_i})


           
        #     lbl_out[mask_i] = iss_class_id
        for idx in range(len(curr_annotations['segments_info'])):
            curr_annotation = curr_annotations['segments_info'][idx]
            annotation_idx = counter.increment()
            if label_mappings[curr_annotation["category_id"]] == 255:
                continue
            coco_ann.append({"id": annotation_idx, "image_id":img_unique_id, "category_id":label_mappings[curr_annotation["category_id"]], "bbox": curr_annotation["bbox"], "iscrowd": curr_annotation["iscrowd"], "area":curr_annotation["area"], "width": self.width, "height":self.height})
            segmInfo.append({"id": curr_annotation["id"],
                             "category_id": curr_annotation["category_id"],
                             "area": curr_annotation["area"],
                             "bbox": curr_annotation["bbox"],
                             "iscrowd": curr_annotation["iscrowd"]})
            # Compute COCO detection format annotation
            # if cs_labels[cls_i].hasInstances:
            #     category_info = {"id": iss_class_id, "is_crowd": iscrowd_i}
            #     coco_ann_i = pct.create_annotation_info(
            #         counter.increment(), img_unique_id, category_info, mask_i, lbl_size, tolerance=2)
            #     if coco_ann_i is not None:
            #         coco_ann.append(coco_ann_i)

        # COCO detection format image annotation
        coco_img = pct.create_image_info(img_unique_id, img_id + ".jpg", lbl_size)

        # Write output
        # Image.fromarray(lbl_out).save(path.join(self.msk_dir, img_id + ".png"))

        # if 'validation' == self.msk_base_dir.split('/')[-2]:
        #     Image.fromarray(lbl).save(path.join(_EVAL, img_id + '.png'))
        #     with Image.open(path.join(self.img_base_dir, img_id + _IMAGE_EXT)) as lbl_img:
        #         img = cv2.resize(np.array(lbl_img), (1280, 384))
        #         Image.fromarray(img).save(path.join(self.img_dir, img_id + ".png"))

        # else: 
        #     shutil.copy(path.join(self.img_base_dir, img_id + _IMAGE_EXT),
        #                 path.join(self.img_dir, img_id + ".png"))

        city_ann = {'image_id': img_id + ".jpg",
                    'file_name': img_id + ".jpg",
                    'segments_info': segmInfo}

        city_img = {"id": img_id,
                    "width": self.width,
                    "height": self.height,
                    "file_name": img_id + ".jpg"}


        return coco_img, coco_ann, city_img, city_ann


def _init_counter(c):
    global counter
    counter = c


class _Counter:
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            val = self.val.value
            self.val.value += 1
        return val


if __name__ == "__main__":
    main(parser.parse_args())
