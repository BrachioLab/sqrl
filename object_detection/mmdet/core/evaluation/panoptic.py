import os
import numpy as np
import json

from . import cityscapes_originalIds
from PIL import Image
import cv2
from shapely.geometry import Polygon

def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)

def calculate_area_after_scaling(mask, scale_factor):
    cor_ls = (np.stack(list(mask.nonzero()),axis=1)/scale_factor).astype(int).tolist()

    return len(set([tuple(cor) for cor in cor_ls]))

def calculate_area_size(poly_ls):
    # obj_to_poly_mappings[pid] = poly_ls
    if len(poly_ls) < 3:
        area = 0

    else:
        polygon=Polygon(poly_ls)

        if polygon.length <= 0:
            area = 0
        else:

            # for poly in poly_ls:
            #     rows_polygon.add((pid, poly[0], poly[1]))

            # "max_x_min_y:number", "max_x_max_y:number", "min_x_min_y:number", "min_x_max_y:number", "max_y_min_x:number", "max_y_max_x:number", "min_y_min_x:number", "min_y_max_x:number"                        # 
            area = polygon.area
    return area

def calculate_area_size_and_select(poly_ls, mask):
    expected_area_size = np.sum(mask)

    min_gap = np.inf
    min_gap_area_size = 0
    min_gap_poly = None

    for poly in poly_ls:
        area_size = calculate_area_size(poly[:,0,:].tolist())
        gap = np.abs(area_size - expected_area_size)

        if gap < min_gap:
            min_gap_area_size = area_size
            min_gap = gap
            min_gap_poly = poly


    x = min_gap_poly[:,0,0].min()
    y = min_gap_poly[:,0,1].min()
    width = min_gap_poly[:,0,0].max() - x + 1
    height = min_gap_poly[:,0,1].max() - y + 1
    return x,y,width, height, min_gap_area_size

def save_panoptic_eval(results, save_dir=None):
    if save_dir is None:
        tmpDir = 'tmpDir'
    else:
        tmpDir = os.path.join(save_dir, "tmpDir")
    createDir(tmpDir)
    base_path = os.path.join(tmpDir, 'tmp')
    base_json = os.path.join(tmpDir, 'tmp_json')
    createDir(base_path)
    createDir(base_json)
    originalIds = cityscapes_originalIds()

    for result in results:
        scale_factor = result[-1][0]['scale_factor']
        images = []
        annotations = []
        pan_pred, cat_pred, meta = result
        pan_pred, cat_pred = pan_pred.numpy(), cat_pred.numpy()
        imgName = meta[0]['filename'].split('/')[-1] 
        if imgName.endswith(".jpg"):
            imageId = imgName.replace(".jpg", "")    
        elif imgName.endswith(".png"):
            imageId = imgName.replace(".png", "")
        inputFileName = imgName
        outputFileName = imgName.replace(".png", "_panoptic.png")
        

        pan_format = np.zeros(
            (pan_pred.shape[0], pan_pred.shape[1], 3), dtype=np.uint8
        )

        panPredIds = np.unique(pan_pred)
        segmInfo = []   

        orig_size = pan_format.shape
        new_size = [v for v in list(orig_size)]
        # scale_factor = 1
        new_size[0] = int(new_size[0]/scale_factor + 0.5)
        new_size[1] = int(new_size[1]/scale_factor + 0.5)

        images.append({"id": imageId,
                       "width": new_size[1],
                       "height": new_size[0],
                       "file_name": inputFileName})
        for panPredId in panPredIds:
            if cat_pred[panPredId] == 255:
                continue
            elif cat_pred[panPredId] <= 10:
                semanticId = segmentId = originalIds[cat_pred[panPredId]] 
            else:
                semanticId = originalIds[cat_pred[panPredId]]
                segmentId = semanticId * 1000 + panPredId 

            isCrowd = 0
            categoryId = semanticId 

            mask = pan_pred == panPredId
            color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
            pan_format[mask] = color


            sub_mask = np.asarray(Image.fromarray(mask).resize([new_size[1], new_size[0]]))

            if np.sum(sub_mask) <= 0:
                pan_format[mask] = 0
                continue

            contours, hierarchy = cv2.findContours(sub_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) <= 0:
                pan_format[mask] = 0
                continue

            x,y,width, height, area = calculate_area_size_and_select(contours, sub_mask)
            # if scale_factor != 1:
            #     area = calculate_area_after_scaling(mask, scale_factor)
            # else:
            # area = np.sum(mask)

            # bbox computation for a segment
            # hor = np.sum(mask, axis=0)
            # hor_idx = np.nonzero(hor)[0]
            # x = hor_idx[0]
            # width = hor_idx[-1] - x + 1
            # vert = np.sum(mask, axis=1)
            # vert_idx = np.nonzero(vert)[0]
            # y = vert_idx[0]
            # height = vert_idx[-1] - y + 1
            # x = x/scale_factor
            # y = y/scale_factor
            # width = width/scale_factor
            # height = height/scale_factor
            # if width <=0 or height <= 0:
            #     pan_format[mask] = 0
            #     continue

            bbox = [int(x), int(y), int(width), int(height)]

            segmInfo.append({"id": int(segmentId),
                             "category_id": int(categoryId),
                             "area": int(area),
                             "bbox": bbox,
                             "iscrowd": isCrowd})
        annotations.append({'image_id': imageId,
                            'file_name': outputFileName,
                            "segments_info": segmInfo})
        

        Image.fromarray(pan_format).resize([new_size[1], new_size[0]]).save(os.path.join(base_path, outputFileName))
        d = {'images': images,
             'annotations': annotations,
             'categories': {}}
        with open(os.path.join(base_json, imageId + '.json'), 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)



def save_bbox_eval(label2cat_mappings, results, meta, save_dir=None):
    if save_dir is None:
        tmpDir = 'tmpDir'
    else:
        tmpDir = os.path.join(save_dir, "tmpDir")
    createDir(tmpDir)
    base_path = os.path.join(tmpDir, 'tmp')
    base_json = os.path.join(tmpDir, 'tmp_json')
    createDir(base_path)
    createDir(base_json)
    originalIds = cityscapes_originalIds()

    result, _ = results

    # for (result,meta) in results:
    imgName = meta['filename'].split('/')[-1] 
    if imgName.endswith(".jpg"):
        imageId = imgName.replace(".jpg", "")    
    elif imgName.endswith(".png"):
        imageId = imgName.replace(".png", "")
    annotations = []
    images = []
    segmInfo = []
    for class_idx in range(len(result)):
        isCrowd = 0
        bbox_ls = result[class_idx]
        for idx in range(bbox_ls.shape[0]):
            bbox = bbox_ls[idx][0:4].tolist()
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            area = bbox[2]*bbox[3]

            segmInfo.append({"category_id": label2cat_mappings[class_idx],
                                "area": int(area),
                                "bbox": bbox,
                                "iscrowd": isCrowd})

    inputFileName = imgName
    outputFileName = imgName.replace(".png", "_panoptic.png")
    annotations.append({'image_id': imageId,
                        'file_name': outputFileName,
                        "segments_info": segmInfo})
    # scale_factor = result[-1][0]['scale_factor']
    
    # annotations = []
    # pan_pred, cat_pred, meta = result
    # pan_pred, cat_pred = pan_pred.numpy(), cat_pred.numpy()
    
    
    

    images.append({"id": imageId,
                    "width": meta['ori_shape'][1],
                    "height": meta['ori_shape'][0],
                    "file_name": inputFileName})
    d = {'images': images,
            'annotations': annotations,
            'categories': {}}
    with open(os.path.join(base_json, imageId + '.json'), 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)

    

