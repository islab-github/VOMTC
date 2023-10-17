# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:43:06 2022

@author: Glen Kim
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 22:31:44 2022

@author: Glen Kim
"""
from collections import OrderedDict
import argparse
import datetime
import os

from scipy.optimize import linear_sum_assignment

import traceback
import cv2

import json
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)



from tqdm.autonotebook import tqdm
from FT_FCN import build_DNN_refiner
#from MMFBM_cnn import CNN_OD, build_CNN_OD # EncoderImageFull



from backbone import EfficientDetBackbone
from efficientdet.dataset import VisionWirelessCroppedDataset
from efficientdet.lossFT_FCN import FocalLossCla
#from efficientdet.lossMMFBM_fixed import FocalLossCla
from efficientdet.utils import BBoxTransform, ClipBoxes

RegressB = BBoxTransform()
ClipB = ClipBoxes()
#from utils.sync_batchnorm import patch_replication_callback
from sync_batchnorm import patch_replication_callback
#from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from utils2 import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from torchvision.ops.boxes import batched_nms
from typing import Union
MODEL_TYPE = 8
PAD_RATIO = 0.3
import torch.nn.functional as F
obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

def Mask(img, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img[y1:y2, x1:x2, : ] = 0
    masked_img = img 
    return masked_img

def HumanMask(img, cropped_mobile_img, box):
    
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    img[:, :, :] = 0 
    img[y1:y2, x1:x2, : ] = cropped_mobile_img
    masked_img = img 
    return masked_img
    
def cropHumanAndMobile(img, box, mobile_box, pad_ratio: float = PAD_RATIO):
    img_h, img_w = img.shape[:2]
    ######human ########
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = (x2 - x1), (y2 - y1)
    pad_w = (w * pad_ratio) // 2
    pad_h = (h * pad_ratio) // 2
    pad = max(pad_w, pad_h)
    new_x1, new_y1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    new_x2, new_y2 = int(min(x2 + pad, img_w)), int(min(y2 + pad, img_h))
    ###### mobile ###########
    x1_mobile, y1_mobile, x2_mobile, y2_mobile = int(mobile_box[0]), int(mobile_box[1]), int(mobile_box[2]), int(mobile_box[3])
    w_mobile, h_mobile = (x2_mobile - x1_mobile), (y2_mobile - y1_mobile)
    pad_w_mobile = (w_mobile * pad_ratio) // 2
    pad_h_mobile = (h_mobile * pad_ratio) // 2
    pad = max(pad_w_mobile, pad_h_mobile)
    new_x1_mobile, new_y1_mobile = int(max(x1_mobile - pad, 0)), int(max(y1_mobile - pad, 0))
    new_x2_mobile, new_y2_mobile = int(min(x2_mobile + pad, img_w)), int(min(y2_mobile + pad, img_h))
    #######################
    
    crop_img = img[new_y1:new_y2, new_x1:new_x2, :].copy()
    
    
    new_x1_mobile = new_x1_mobile - new_x1
    new_y1_mobile = new_y1_mobile - new_y1
    new_x2_mobile = new_x2_mobile - new_x1
    new_y2_mobile = new_y2_mobile - new_y1
    crop_mobile_img = crop_img[new_y1_mobile:new_y2_mobile, new_x1_mobile:new_x2_mobile, :].copy()
    
    return crop_img, (new_x1, new_y1, new_x2, new_y2),  crop_mobile_img, (new_x1_mobile, new_y1_mobile, new_x2_mobile, new_y2_mobile)
    
def cropHumanAndMobile2D(img, box, mobile_box, pad_ratio: float = PAD_RATIO):
    img_h, img_w = img.shape
    ######human ########
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = (x2 - x1), (y2 - y1)
    pad_w = (w * pad_ratio) // 2
    pad_h = (h * pad_ratio) // 2
    pad = max(pad_w, pad_h)
    new_x1, new_y1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    new_x2, new_y2 = int(min(x2 + pad, img_w)), int(min(y2 + pad, img_h))
    ###### mobile ###########
    x1_mobile, y1_mobile, x2_mobile, y2_mobile = int(mobile_box[0]), int(mobile_box[1]), int(mobile_box[2]), int(mobile_box[3])
    w_mobile, h_mobile = (x2_mobile - x1_mobile), (y2_mobile - y1_mobile)
    pad_w_mobile = (w_mobile * pad_ratio) // 2
    pad_h_mobile = (h_mobile * pad_ratio) // 2
    pad = max(pad_w_mobile, pad_h_mobile)
    new_x1_mobile, new_y1_mobile = int(max(x1_mobile - pad, 0)), int(max(y1_mobile - pad, 0))
    new_x2_mobile, new_y2_mobile = int(min(x2_mobile + pad, img_w)), int(min(y2_mobile + pad, img_h))
    #######################
    
    crop_img = img[new_y1:new_y2, new_x1:new_x2].copy()
    
    
    new_x1_mobile = new_x1_mobile - new_x1
    new_y1_mobile = new_y1_mobile - new_y1
    new_x2_mobile = new_x2_mobile - new_x1
    new_y2_mobile = new_y2_mobile - new_y1
    crop_mobile_img = crop_img[new_y1_mobile:new_y2_mobile, new_x1_mobile:new_x2_mobile].copy()
    
    return crop_img, (new_x1, new_y1, new_x2, new_y2),  crop_mobile_img, (new_x1_mobile, new_y1_mobile, new_x2_mobile, new_y2_mobile)
  
def crop(img, box, pad_ratio: float = PAD_RATIO):
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = (x2 - x1), (y2 - y1)
    pad_w = (w * pad_ratio) // 2
    pad_h = (h * pad_ratio) // 2
    pad = max(pad_w, pad_h)
    new_x1, new_y1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    new_x2, new_y2 = int(min(x2 + pad, img_w)), int(min(y2 + pad, img_h))

    crop_img = img[new_y1:new_y2, new_x1:new_x2, :].copy()
    return crop_img, (new_x1, new_y1, new_x2, new_y2)

def crop2D(img, box, pad_ratio: float = PAD_RATIO):
    img_h, img_w = img.shape #[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = (x2 - x1), (y2 - y1)
    pad_w = (w * pad_ratio) // 2
    pad_h = (h * pad_ratio) // 2
    pad = max(pad_w, pad_h)
    new_x1, new_y1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    new_x2, new_y2 = int(min(x2 + pad, img_w)), int(min(y2 + pad, img_h))

    crop_img = img[new_y1:new_y2, new_x1:new_x2].copy()
    return crop_img, (new_x1, new_y1, new_x2, new_y2)

def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h, x1, y1, x2, y2  = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds

def aspectaware_resize_padding_updated(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height
        
    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    #
    # x1, y1, x2, y2: coordinates of mobile on framed_img
    #
    index_set = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]): 
            if(image[i,j,0] != image[0,0,0]):
                index_set.append((i, j))
    x1, y1 = index_set[0]
    x2, y2 = index_set[len(index_set)-1]
    
    
    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h, x1, y1, x2, y2


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    
    new_w = width
    new_h = height
    #if old_w > old_h:
    #    new_w = width
    #    new_h = int(width / old_w * old_h)
    #else:
    #    new_w = int(height / old_h * old_w)
    #    new_h = height
    #canvas = np.zeros((height, height, c), np.float32)
    canvas = np.zeros((height, width, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
        
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image
    
    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,

def detect(framed_metas, model, img, threshold: float, iou_threshold: float):
    #EfficientNet 통과해서 총 4개 나옴
    features, regression, classification, anchors = model(img)
    
    
    
    #print('-------regression--------')
    #print(regression.shape) # (4, 442260, 4)
    #print('-------classification------')
    #print(classification.shape) #(4, 442260, 90)
    #print('-------anchors------')
    #print(anchors.shape) #(4, 442260, 4)
    
    shapex = regression.shape[0]
    out = postprocess(shapex,
        img,
        anchors, regression, classification,
        RegressB, ClipB,
        threshold, iou_threshold
    ) #rois, class_ids, scores
    
    
    out = invert_affine(framed_metas, out)
    
    
    return out


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:2")
device3 = torch.device("cuda:3")

def scatter1(inputs):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
    num_gpus = 4
    #devices = ['():' + str(x) for x in range(1, num_gpus)]
    splits = inputs.shape[0] // num_gpus



    
        
    hey = [((inputs[(splits * device_idx) : (splits * (device_idx + 1))]).to(f'cuda:{device_idx}', non_blocking=True).detach().cpu().numpy())
            for device_idx in range(1,num_gpus)] #.detach().cpu().numpy()
            
    return torch.tensor(hey[0]).squeeze(0).to(device1)   , torch.tensor(hey[1]).squeeze(0).to(device2), torch.tensor(hey[2]).squeeze(0).to(device3)



def parse_prediction(pred: dict, target: str = "phone", threshold: float = 0.0):
    assert target in ("person", "phone")
    result = []
    for v in pred.values():
        if v["score"] < threshold * 100:
            continue

        if ("phone" in v["obj"]) and (target == "phone"):
            box = [v["x1"], v["y1"], v["x2"], v["y2"]]
            result.append(box)
        elif ("person" in v["obj"]) and (target == "person"):
            box = [v["x1"], v["y1"], v["x2"], v["y2"]]
            result.append(box)
    return result

def postprocess6(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold): #threshold:0.01
    transformed_anchors = regressBoxes(anchors, regression) #.detach().cpu()
    transformed_anchors = clipBoxes(transformed_anchors, x) #.detach().cpu()
    
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    
    scores_over_thresh = (scores > threshold)[:, :, 0]
    
    
   
    out = []
    true_indices_set = [[]]*x.shape[0]
    
      
        
    for i in range(x.shape[0]): #images 갯수
        
        true_indices = []
        scores_over_thresh0, scores_over_thresh1, scores_over_thresh2 = scatter1(scores_over_thresh[i,:])
        
        #print('scores over thresh0') #(110565)
        #print(np.shape(scores_over_thresh0))
        
        #print('scores over thresh1')
        #print(np.shape(scores_over_thresh1))
       
        #print('scores over thresh2')
        #print(np.shape(scores_over_thresh2))
        
        #print('scores over thresh3')
        #print(np.shape(scores_over_thresh3))
        
        true_indices = func0(scores_over_thresh0, true_indices) #.to(device0)
        true_indices = func1(scores_over_thresh1, true_indices) #.to(device1)
        true_indices = func2(scores_over_thresh2, true_indices) #.to(device2)
        #.to(device3)
        
        #print('INDICES_OVER_THRESH')
        #print(len(true_indices)) 
        true_indices_set[i].append(true_indices)
        
        if scores_over_thresh[i].sum() == 0: #threshold를 넘긴 scores의 합이 0이면, empty array가 넣어짐
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue
        #각 image마다 classification 점수들
        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        
        
        
        #CLA_PER = np.concatenate([np.reshape(classification_per[0,:], [1,np.shape(classification_per[0,:])[0]]), np.reshape(classification_per[76,:], [1,np.shape(classification_per[76,:])[0]])], axis = 0)
        
        #각 image마다 transformed anchor
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        
        #각 image마다 score
        #scores_per = scores[i, scores_over_thresh[i, :], ...]
        
        #각 image마다 classification 점수들의 max인 점수와 해당 class
        scores_, classes_ = classification_per.max(dim=0)
        
        out.append({'rois': transformed_anchors_per.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
        })
    
        #print(np.shape(true_indices_set))
    true_indices_set = np.array(true_indices_set) #(1, 1, 2030)
    true_indices_set = np.reshape(true_indices_set, [np.shape(true_indices_set)[2]])
    #print(np.shape(true_indices_set)) #(2030,)
    return out, true_indices_set

### 너무 느림 :(
def postprocess5(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold): #threshold:0.01
    transformed_anchors = regressBoxes(anchors, regression) #.detach().cpu()
    transformed_anchors = clipBoxes(transformed_anchors, x) #.detach().cpu()
    
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    true_indices_set = [[]]*x.shape[0]
    
      
        
    for i in range(x.shape[0]): #images 갯수
        
        true_indices = []
        for ii in range(scores_over_thresh[i,:].shape[0]):
            if(scores_over_thresh[i, :][ii] == True):
                true_indices.append(ii)
        #print('INDICES_OVER_THRESH')
        #print(len(true_indices)) 
        true_indices_set[i].append(true_indices)
        
        if scores_over_thresh[i].sum() == 0: #threshold를 넘긴 scores의 합이 0이면, empty array가 넣어짐
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue
        #각 image마다 classification 점수들
        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        
        
        
        #CLA_PER = np.concatenate([np.reshape(classification_per[0,:], [1,np.shape(classification_per[0,:])[0]]), np.reshape(classification_per[76,:], [1,np.shape(classification_per[76,:])[0]])], axis = 0)
        
        #각 image마다 transformed anchor
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        
        #각 image마다 score
        #scores_per = scores[i, scores_over_thresh[i, :], ...]
        
        #각 image마다 classification 점수들의 max인 점수와 해당 class
        scores_, classes_ = classification_per.max(dim=0)
        
        out.append({'rois': transformed_anchors_per.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
        })
    
        #print(np.shape(true_indices_set))
    true_indices_set = np.array(true_indices_set) #(1, 1, 2030)
    true_indices_set = np.reshape(true_indices_set, [np.shape(true_indices_set)[2]])
    #print(np.shape(true_indices_set)) #(2030,)
    return out, true_indices_set

def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]): #images 갯수
        if scores_over_thresh[i].sum() == 0: #threshold를 넘긴 scores의 합이 0이면, empty array가 넣어짐
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue
        #각 image마다 classification 점수들
        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        
        #각 image마다 transformed anchor
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        
        #각 image마다 score
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        
        #각 image마다 classification 점수들의 max인 점수와 해당 class
        scores_, classes_ = classification_per.max(dim=0)
        scores_person = classification_per[0, :]
        scores_phone = classification_per[76, :]
        #442260
        #performs nms in a batch fashion.
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)
        
        #
        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]
            
            
            scores_person = scores_person[anchors_nms_idx]
            scores_phone = scores_phone[anchors_nms_idx]
            
            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
                'scores_person': scores_person.cpu().numpy(),
                'scores_phone': scores_phone.cpu().numpy(),
            })
    
        # 만약 anchors_nms_idx.shape[0] == 0이라면:
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'scores_person': np.array(()),
                'scores_phone': np.array(()),
            })
        

    return out



def parse_distance(json_path: str) -> np.ndarray:
    with open(json_path, "r") as f:
        dist = json.load(f)

    # key = "y_x" , y [0, 480), x [0, 640)
    result = np.zeros((480, 640), dtype=np.float32)
    for k, v in dist.items():
        y, x = k.split("_")
        result[int(y), int(x)] = float(v)
    return result

def preprocess(image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, 819, 540,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

def preprocess_UPDATED(image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    #ori_imgs = image_path
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    # ori_imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in image_path]
    
    
    #각 이미지마다 normalize
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    
    #각 normalized 이미지마다 aspectaware resize padding
    imgs_meta = [aspectaware_resize_padding_updated(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    
    #각 imgs_meta 이미지마다 0번째 index에 있는 값은 framed_img
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    
    #각 img_meta 이미지마다 나머지 index에 있는 값들은 framed_metas
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

def two_detect_version2(framed_metas, img1, anchors, regression, classification, threshold:float, iou_threshold:float):
    
    out, true_indices_set = postprocess6(
        img1,
        anchors, regression, classification,
        RegressB, ClipB,
        threshold, iou_threshold
    )
    
    #rois, class_ids, scores
    
    
    out = invert_affine(framed_metas, out)
    
    return out, true_indices_set

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)
    
def bbox_iou(box1, box2) -> float:
    box1_area = (box1[:,2] - box1[:,0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    x1_max = max(box1[:, 0], box2[:, 0])
    y1_max = max(box1[:, 1], box2[:, 1])
    x2_min = min(box1[:, 2], box2[:, 2])
    y2_min = min(box1[:, 3], box2[:, 3])

    inter_x = x2_min - x1_max + 1
    inter_y = y2_min - y1_max + 1

    if (inter_x <= 0) or (inter_y <= 0):
        return -1.0

    inter = inter_x * inter_y
    union = box1_area + box2_area - inter
    iou = float(inter / union)
    return iou

def match_bboxes(bbox_gt, bbox_pred, threshold=0.3):
    """
    https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2].
      The number of bboxes, N1 and N2, need not be the same.
    threshold: IOU threshold

    Returns
    -------
    (idxs_true, idxs_pred, ious)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
    """
    if (len(bbox_gt) == 0) or (len(bbox_pred) == 0):
        return [], [], []

    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i], bbox_pred[j])

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate((iou_matrix, np.zeros((diff, n_pred))), axis=0)
    elif n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate((iou_matrix, np.zeros((n_true, diff))), axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = linear_sum_assignment(1 - iou_matrix)

    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > threshold)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid]


def non_max_suppression_fast(boxes, scores=None, iou_threshold: float = 0.1):
    """
    boxes : coordinates of each box
    scores : score of each box
    iou_threshold : iou threshold(box with iou larger than threshold will be removed)
    """
    if len(boxes) == 0:
        return []
    if len(boxes) == 1:
        return boxes

    # Init the picked box info
    pick = []

    # Box coordinate consist of left top and right bottom
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute area of each boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Greedily select the order of box to compare iou
    if scores is None:
        scores = np.ones((len(boxes),), dtype=np.float32)
    indices = np.argsort(scores)

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[0]
        pick.append(i)

        # With vector implementation, we can calculate fast
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h

        # Calculate the iou
        iou = intersection / (area[indices[:last]] + area[indices[last]] - intersection)
        
        # if IoU of a box is larger than threshold, remove that box index.
        indices = np.delete(indices, np.concatenate(([last], np.where(iou > iou_threshold)[0])))

    pick = list(set(pick))
    return boxes[pick]


def load_prediction(json_path: str = "data/prediction.json"):
    with open(json_path, "r") as f:
        pred = json.load(f)

    pred = OrderedDict(sorted(pred.items()))
    print(f"Prediction loaded, total: {len(pred)} predictions")
    return pred


        
        
PredJson = "/home/glen/EfficientDet/predictionBATCH.json"       
def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='VisionWireless', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=8, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=4               )
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=10, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='/home/glen/VisionWireless/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='/home/glen/EfficientDet/logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--load_weights_FTFCN', type=str, default=None,
                        help='whether to load weights from a FTFCN checkpoint')                        
    parser.add_argument('--saved_path', type=str, default='/home/glen/EfficientDet/logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args

phase = 'train'
size = 512
num_classes = 2

class ModelCNN(nn.Module):
    def __init__(self, model1,  debug=False):
        super().__init__()
        self.criterion = FocalLossCla()
        self.debug = debug
        #self.fc = nn.Linear(, )
        self.model1 = model1
        
        
        #self.init_weights()
        
    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        
    def forward(self,  depth_imgs):
        #
        # INPUT MUST BE PADDED
        # 
        #
        
        depth_img_features = self.model1(depth_imgs)
        
        return depth_img_features
    
class ModelWithLoss(nn.Module):
    def __init__(self,  model2, debug=False):
        super().__init__()
        self.criterion = FocalLossCla()
        self.debug = debug
        #self.fc = nn.Linear(, )
        
        self.model2 = model2
        
        #self.init_weights()
        
    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        
    def forward(self, framed_metas, classification_f, classification_true, lengths):
        #
        # INDEXs 사실상 안쓰임 ㅋㅋ
        # 
        #
   
        classification_resized = torch.reshape(classification_f, [classification_f.shape[0], classification_f.shape[1]* classification_f.shape[2]])
        classification_resized = classification_resized.detach().cpu()
        
        
        
        classification_input = classification_resized
        
        
        
        #
        # Multi-Modality Mixer
        #
        classification_output = self.model2(classification_input.to(device2))
        
        
        
        classification_output = torch.reshape(classification_output, [classification_output.shape[0], int(classification_output.shape[1]/2), 2]).to(device1)
        torch.cuda.empty_cache()
        
        Final_classification_output = torch.empty(classification_output.shape[0], 200, 2)
        
        for ii in range(classification_output.shape[0]):
            classification_output_perbatch = classification_output[ii, :,:]
            
            classification_output_perbatch = F.softmax(classification_output_perbatch)
            
            
            #classification_output_perbatch = classification_output_perbatch
            Final_classification_output[ii, :, :] =  classification_output_perbatch.unsqueeze(0)
            
        Final_classification_output = Final_classification_output.to(device1)
        classification_true = classification_true.to(device1)
        
        #print('EXPECTED')
        #print('[8, 50, 2]')
        #print('REAL')
        #print(Final_classification_output.shape)
        
        
        #print('classification_true')
        #print('should be [8, 50, 2]')
        #print(classification_true.shape)
        
        
        
        
        #NONOVERLAPPING_INDICES_SET = []
        #CLASS_IDs = []
        #for jjjj in range(classification.shape[0]):
            
            
            
        #    classID = CLASSIDs[jjjj]
        #    index = INDEXs[jjjj]
             
            
        #    NONOVERLAPPING_ROIS_indexes = np.array(np.load(index))
        #    classIDs = np.array(np.load(classID))
         
        #    NONOVERLAPPING_INDICES_SET.append(NONOVERLAPPING_ROIS_indexes)
        #    CLASS_IDs.append(classIDs)
            
        
        
        #NONOVERLAPPING_INDICES_SET = np.array(NONOVERLAPPING_INDICES_SET, dtype = object)
        #CLASS_IDs = np.array(CLASS_IDs, dtype = object)
      
        cls_loss1 = self.criterion(Final_classification_output, classification_true, lengths)
        return cls_loss1
    
def adjust_learning_rate(opt, optimizer, epoch, lr_schedules):
    #logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        #logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        print('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))        
            
    
def load_model(model_type: int = MODEL_TYPE):
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)] #(1, 1.259921, 1.5874)
    model = EfficientDetBackbone(
        compound_coef=model_type, num_classes=90,
        anchor_ratios=anchor_ratios, anchor_scales=anchor_scales
    )
    model.load_state_dict(torch.load(f"logs/coco/efficientdet-d{model_type}.pth", map_location="cpu"), strict=True)
    
    #model = CustomDataParallel(model, 4)
    model.eval()
    return model, input_sizes[model_type]    

def train(opt):
    params = Params(f'projects/{opt.project}.yml')
    
    
    #if params.num_gpus == 0:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'num_workers': opt.num_workers}
                       
    #'collate_fn': collater,
                       

    #val_params = {'batch_size': opt.batch_size,
    #              'shuffle': False,
    #              'drop_last': True,
    #              'collate_fn': collater,
    #              'num_workers': opt.num_workers}

    
    
    training_set = VisionWirelessCroppedDataset(root_dir=os.path.join(opt.data_path, params.project_name), 
                                         transform=None)
    
    training_generator = DataLoader(training_set, **training_params)
    
    
    #training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
    #                           imageset = 'images/train2014/',
    #                           transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
    #                                                         Augmenter(),
    #                                                         Resizer(input_sizes[opt.compound_coef])]))
    #training_generator = DataLoader(training_set, **training_params)

    #val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
    #                      imageset = 'images/val2014/',
    #                      transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
    #                                                    Resizer(input_sizes[opt.compound_coef])]))
    #val_generator = DataLoader(val_set, **val_params)

    det_model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    #det_model, input_size = load_model(model_type=MODEL_TYPE)
    
    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0
        try:
            ret = det_model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
           print(f'[Warning] Ignoring {e}')
           print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
       last_step = 0
       print('[Info] initializing weights...')
       init_weights(det_model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN', 'Regressor', 'Classifier']:
                if ntl in classname:
                   for param in m.parameters():
                       param.requires_grad = False

        det_model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    
    #gpo1 = GPO(opt.batch_size, opt.batch_size)
    #gpo2 = GPO_Cla(opt.batch_size)
    
    #model1 = build_CNN_OD(phase, size, num_classes)
    model2 = build_DNN_refiner()
    
    ################################################################################################
    # Option to resume at a checkpoint
    ################################################################################################
    
   
    
    print('-------------------FROM CHECKPOINT----------------------------')
    state_dict_path2 =  '/home/glen/EfficientDet/logs/VisionWireless/FTFCN-d8_0.0001_0_283_BATCH32.pth'


    

    loaded_state_dict2 = torch.load(state_dict_path2)
    new_state_dict2 = OrderedDict()
    for n, v in loaded_state_dict2.items():
        name = n.replace("model2.","") 
        new_state_dict2[name] = v

    model2.load_state_dict(new_state_dict2)

    print('MODELS LOADED')
    
    
    ##########################################################
    
    

    torch.set_grad_enabled(True)
    #model1.train()
    model2.train()
    ## RESUME FROM CHECKPOINT fusionmodel2.init_MMFBM
    
    
    
    #if opt.load_weights_MMFBM is None:
        
    #    state_dict_path = '/home/glen/EfficientDet/logs/VisionWireless/poolingCla_AncAvg-d8_0.0001_88_84300_changedModelCROPtraining_BATCH4_ClaConvAuto.pth'
    #    loaded_state_dict = torch.load(state_dict_path)
    #    new_state_dict = OrderedDict()
    #    for n, v in loaded_state_dict.items():
    #        name = n.replace("fusionmodel2.","") 
    #        new_state_dict[name] = v

    #    fusionmodel2.load_state_dict(new_state_dict)
                
                
                
    #fusionmodel2.train()
    
    
    
    
    ######################################################
    #fus_model1 = ModelCNN(model1, debug=opt.debug)
    fus_model2 = ModelWithLoss( model2, debug=opt.debug)
    
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        det_model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    
    if params.num_gpus > 0:
        det_model = det_model.cuda()
        
        if params.num_gpus > 1:
            det_model = CustomDataParallel(det_model, params.num_gpus)
            
        if use_sync_bn:
    #    patch_replication_callback(fus_model)
           patch_replication_callback(det_model)
          
    
           
    params = list(model2.parameters())
    #params =  list(fus_model2.parameters())
    #params = list(model1.parameters())
    #lr_schedules = [opt.lr_update, ]
    if opt.optim == 'adamw':
        optimizer1 = torch.optim.AdamW(params, opt.lr)
        #optimizer2 = torch.optim.AdamW(fus_model2.parameters(), opt.lr)
        
    else:
        optimizer1 = torch.optim.SGD(params, opt.lr, momentum=0.9, nesterov=True)
        #optimizer2 = torch.optim.SGD(fus_model2.parameters(), opt.lr, momentum=0.9, nesterov=True)
        

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, factor = 0.1, patience=2, mode ='min', verbose = True)
    #scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, patience=3, verbose=True)
    
    
    last_step = 0
    
    #f1score = 0
    #best_f1score = 0
    #print(f'resuming checkpoint from step: {last_step}')
    #print('MODELS LOADED')
    step = max(0, last_step)
    
    
    num_iter_per_epoch = len(training_generator)
    
    try:
        for epoch in range(1, opt.num_epochs):
        #for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue
            
            
            train_losses = []
            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    
                
                    #imgs = data['img']
                    
                    #
                    # 여기서 말하는 rgb_path는 cropped RGB path
                    #
                    rgb_imgs = data['rgb_path']
                    #depth_imgs = data['depth_path']
                    CLASSIDs = data['CLASSID']
                    INDEXs = data['INDEX']
                    ROISs = data['ROIS']
                    SCOREs = data['SCORE']
                    
                    
                    
                    
                    #print(depth_imgs.shape) #(4, 480, 640)
                    #if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        
                        
                    #imgs = imgs.cuda()
                    #depth_imgs = depth_imgs.cuda()
                    
                    
                    
                    
                    #annot = annot.cuda()

                    optimizer1.zero_grad()
                    #optimizer2.zero_grad()
                    
                    #_, regression, classification, anchors = det_model(imgs)
                    #_, regression_depth, classification_depth, anchors_depth = det_model(depth_imgs)
                    
                    input_size = 1536
                    #ori_img1, framed_img1, framed_metas = preprocess_UPDATED(rgb_path, max_size=input_size)
                    rgb_imgs = np.array(rgb_imgs)
                    #depth_imgs = np.array(depth_imgs)
                    #print('---------framed_img1---------')
                    #print(len(ori_img1)) 
                    
                    #print(len(ori_img2))
                    #print('ori img 2')
                    #print(ori_img2[0].shape) #(480, 640, 3)
                    
                    cls_loss1_tot = 0
                    model = det_model.eval()
                    #reg_loss_tot = 0
                    
                    #concatclassification =  torch.empty([1, 442260, 90])
                    #concatregression = torch.empty([1, 442260, 4])
                    #concatanchors = torch.empty([1, 442260, 4])
                    #concatregression_depth = torch.empty([1, 442260, 4])
                    #concatclassification_depth = torch.empty([1, 442260, 90])
                    #concatanchors_depth = torch.empty([1, 442260, 4])
                    
                    
                    
                    #concatimages = torch.empty([1, 3, input_size, input_size]).cuda()
                    #for i in range(len(framed_img1)):
                    #    img_th = torch.from_numpy(framed_img1[i]).cuda().unsqueeze(0).permute(0, 3, 1, 2)
                    #    concatimages = torch.cat([concatimages, img_th], 0).cuda()
                    #concatimages = concatimages[1:]
                    #print('---------concatimages---------')
                    #print(concatimages.shape) #(6, 3, 1536, 1536)
                    
                    
                    ############# LARGE-SCALE OBJECT DETECTION ####################
                    #crop_paths1 = []
                    #crop_paths2 = []
                    
                    #for iii in range(len(framed_img1)):
                       
                    #    prediction = detect(framed_metas, model, concatimages, threshold=0.4, iou_threshold=0.4)[iii]
                    #    for j in range(len(prediction['rois'])):
                    #        obj = obj_list[prediction['class_ids'][j]]
            
                            ## 1st stage: person detection
                    #        if obj == "person":
                    #            x1, y1, x2, y2 = prediction['rois'][j].astype(np.int32)
                       
                    #            crop_img1, (crop_x1, crop_y1, crop_x2, crop_y2) = crop(ori_img1[iii], (x1, y1, x2, y2), pad_ratio=PAD_RATIO)
                    #            crop_img2, _ = crop2D(depth_imgs[iii].numpy(), (x1, y1, x2, y2), pad_ratio=PAD_RATIO)
                            
                    #            crop_path1 = "/home/glen/EfficientDet/crop_RGB/crop_RGB" + str(iii) + '_' + str(j) + ".png"
                    #            crop_path2 = "/home/glen/EfficientDet/crop_DEPTH/crop_DEPTH" + str(iii) + '_' + str(j) + ".png"
                    #            cv2.imwrite(crop_path1, crop_img1)
                
                    #            plt.imsave(crop_path2, crop_img2)
                    #    crop_paths1.append(crop_path1)
                    #    crop_paths2.append(crop_path2)
                        #### BATCH SIZE ####
                    #_, framed_img1, framed_metas = preprocess(crop_paths1, max_size = input_size)
                    #_, framed_img2, _ = preprocess(crop_paths2, max_size = input_size)
                    ###########################OBJECT DETECTOR 사용하지 않구 ########################################
                    
                    
                    
                    #final_crop_coordinates = np.zeros((1,4), dtype = np.int32)
                    # 각 CROP 이미지마다
                        ##########################################
                                    
                            
                        #final_crop_coordinates =  np.concatenate([final_crop_coordinates, np.reshape(np.array([crop_x1, crop_y1, crop_x2, crop_y2]), [1,4])], axis = 0)
                        #print(final_phone_annotations)
                    #final_crop_coordinates = final_crop_coordinates[1:]
                    
                            #final_crop_coordinates = np.delete(final_crop_coordinates, zz)
                    #print('PHONE ANNOTATIONS and CROP PATHS LENGTH')
                    #print(np.shape(final_phone_annotations))
                    #print('CROP PATHS LENGTH')
                    #print(len(crop_paths1))
                    #print(len(crop_paths2))
                    #print(crop_paths2)
                        
                    _, framed_img1, framed_metas = preprocess_UPDATED(rgb_imgs, max_size = input_size)
                    #_, framed_img2, _ = preprocess_UPDATED(crop_paths2, max_size = input_size)
                    #depth_img_size = 512
                    
                    #_, framed_img2, _ = preprocess_UPDATED(depth_imgs, max_size = input_size)
                        
                    
                    #print('framed_img2')
                    #print(framed_img2[0].shape) #(540, 819, 3)
                    
                    ####################################################################
                    concatimages1 = torch.empty([1, 3, input_size, input_size]).cuda()
                    concatimages1_1 = torch.empty([1, 3, input_size, input_size]).to(device1)
                    concatimages1_2 = torch.empty([1, 3, input_size, input_size]).to(device2)
                    concatimages1_3 = torch.empty([1, 3, input_size, input_size]).to(device3)
                    #concatimages2 = torch.empty([1, 3, 540, 819]).to(device3)
                    for ii in range(0, 8):
                        img_th1 = torch.from_numpy(framed_img1[ii]).cuda().unsqueeze(0).permute(0, 3, 1, 2)
                        concatimages1 = torch.cat([concatimages1, img_th1], 0).cuda()
                    
                    
                        #img_th2 = torch.from_numpy(framed_img2[ii]).to(device3).unsqueeze(0).permute(0, 3, 1, 2)
                        #concatimages2 = torch.cat([concatimages2, img_th2], 0).to(device3)
                    for iii in range(8, 16):
                        img_th1_1 = torch.from_numpy(framed_img1[iii]).to(device1).unsqueeze(0).permute(0, 3, 1, 2)
                        concatimages1_1 = torch.cat([concatimages1_1, img_th1_1], 0).to(device1)
                        
                    for iiii in range(16, 24):
                        img_th1_2 = torch.from_numpy(framed_img1[iiii]).to(device2).unsqueeze(0).permute(0, 3, 1, 2)
                        concatimages1_2 = torch.cat([concatimages1_2, img_th1_2], 0).to(device2)
                        
                    for iiiii in range(24, len(framed_img1)):
                        img_th1_3 = torch.from_numpy(framed_img1[iiiii]).to(device3).unsqueeze(0).permute(0, 3, 1, 2)
                        concatimages1_3 = torch.cat([concatimages1_3, img_th1_3], 0).to(device3)
                    concatimages1 = concatimages1[1:]
                    concatimages1_1 = concatimages1_1[1:]
                    concatimages1_2 = concatimages1_2[1:]
                    concatimages1_3 = concatimages1_3[1:]
                    #concatimages2 = concatimages2[1:]
                    ####################################################################
                    concatimagesfinal = torch.cat([concatimages1.detach().cpu(), concatimages1_1.detach().cpu(), concatimages1_2.detach().cpu(), concatimages1_3.detach().cpu()], 0)
                    ###########################################################
                    
                    
                        
                    
                            
                                       
                    #print('-----concatimages1-----')
                    #print(concatimages1.shape) #(8,3,1536, 1536) #8보다 클수 있음
                    #print('-----------------------')
                    #print('-----concatimages2-----')
                    #print(concatimages2.shape) #(7, 3, 1536, 1536)
                    
                        
                    #elif concatimages1.shape[0] > 8 and concatimages1.shape[0] <= 10: #12 
                    #    _, regression_A, classification_A, anchors_A = model(concatimages1[0:4])
                    #    _, regression_depth_A, classification_depth_A, anchors_depth_A = model(concatimages2[0:4])
                        
                    #    _, regression_B, classification_B, anchors_B = model(concatimages1[4:8])
                    #    _, regression_depth_B, classification_depth_B, anchors_depth_B = model(concatimages2[4:8])
                        
                        #2 
                    #    lastImagesSize = len(concatimages1[8:])
                    #    _, regression_C, classification_C, anchors_C = model(torch.cat([concatimages1[8:], torch.empty([4 - len(concatimages1[8:]) ,3,input_size, input_size]).cuda() ], axis = 0))
                    #    _, regression_depth_C, classification_depth_C, anchors_depth_C = model(torch.cat([concatimages1[8:], torch.empty([4 - len(concatimages1[8:]) ,3,input_size, input_size]).cuda() ], axis = 0))
                        
                        #### 4, 4, 2
                    #    regression = torch.cat([regression_A, regression_B, regression_C[0:lastImagesSize]], axis = 0)
                    #    classification = torch.cat([classification_A, classification_B, classification_C[0:lastImagesSize]], axis = 0)
                    #    anchors = torch.cat([anchors_A, anchors_B, anchors_C[0:lastImagesSize]], axis = 0)
                        
                    #    regression_depth = torch.cat([regression_depth_A, regression_depth_B, regression_depth_C[0:lastImagesSize]], axis = 0)
                    #    classification_depth = torch.cat([classification_depth_A, classification_depth_B, classification_depth_C[0:lastImagesSize]], axis = 0)
                    #    anchors_depth = torch.cat([anchors_depth_A, anchors_depth_B, anchors_depth_C[0:lastImagesSize]], axis = 0)
                        
                   
                        
                        #final_phone_annotations = final_phone_annotations[0:6]
                    
                   
                    if concatimages1.shape[0] == 8 and concatimages1_1.shape[0] == 8:
                    # OBJECT DETECT
                        
                        _, regression_A, classification_A, anchors_A = model(concatimages1[0:4])
                        
                        
                        
                        _, regression_B, classification_B, anchors_B = model(concatimages1[4:8])
                        
                        
                        _, regression_C, classification_C, anchors_C = model(concatimages1_1[0:4])
                        
                        
                        _, regression_D, classification_D, anchors_D = model(concatimages1_1[4:8])
                        
                        
                        _, regression_E, classification_E, anchors_E = model(concatimages1_2[0:4])
                        
                        _, regression_F, classification_F, anchors_F = model(concatimages1_2[4:8])
                        
                        
                        
                        
                        _, regression_G, classification_G, anchors_G = model(concatimages1_3[0:4])
                        
                        _, regression_H, classification_H, anchors_H = model(concatimages1_3[4:8])
                        
                        
                        
                        
                        #2 
                        #lastImagesSize = 2
                        #_, regression_C, classification_C, anchors_C = model(torch.cat([concatimages1[8:10], torch.empty([lastImagesSize ,3,input_size, input_size]).cuda() ], axis = 0))
                        #_, regression_depth_C, classification_depth_C, anchors_depth_C = model(torch.cat([concatimages1[8:10], torch.empty([lastImagesSize ,3,input_size, input_size]).cuda() ], axis = 0))
                        
                        #### 4, 4, 2
                        regression = torch.cat([regression_A.detach().cpu(), regression_B.detach().cpu(), regression_C.detach().cpu(), regression_D.detach().cpu(), regression_E.detach().cpu(), regression_F.detach().cpu(), regression_G.detach().cpu(), regression_H.detach().cpu()], axis = 0)
                        classification = torch.cat([classification_A.detach().cpu(), classification_B.detach().cpu(), classification_C.detach().cpu(), classification_D.detach().cpu(), classification_E.detach().cpu(), classification_F.detach().cpu(), classification_G.detach().cpu(), classification_H.detach().cpu()], axis = 0)
                        anchors = torch.cat([anchors_A.detach().cpu(), anchors_B.detach().cpu(), anchors_C.detach().cpu(), anchors_D.detach().cpu(), anchors_E.detach().cpu(), anchors_F.detach().cpu(), anchors_G.detach().cpu(), anchors_H.detach().cpu()], axis = 0)
                        
                        
                        
                        #_, regression_A, classification_A, anchors_A = model_OD1(concatimages1[0:4].to(device0))
                        
                        
                        
                        #_, regression_B, classification_B, anchors_B = model_OD1(concatimages1[4:8].to(device0))
                        
                        
                        #_, regression_C, classification_C, anchors_C = model_OD2(concatimages1[8:12].to(device1))
                        
                        
                        
                      
                        
                        #### 4, 4, 2
                        #regression = torch.cat([regression_A, regression_B, regression_C.cuda(), regression_D.cuda(), regression_E.cuda(), regression_F.cuda(), regression_G.cuda(), regression_H.cuda()], axis = 0)
                        
                        #classification = torch.cat([classification_A, classification_B, classification_C.cuda(), classification_D.cuda(), classification_E.cuda(), classification_F.cuda(), classification_G.cuda(), classification_H.cuda()], axis = 0)
                        
                        #anchors = torch.cat([anchors_A, anchors_B, anchors_C.cuda(), anchors_D.cuda(), anchors_E.cuda(), anchors_F.cuda(), anchors_G.cuda(), anchors_H.cuda()], axis = 0)
                        
                        
                        
                        #classificationAA = classification[:, :, 0].unsqueeze(2)
                        #classificationBB = classification[:, :, 76].unsqueeze(2)
                        ####### 
        
                        #Human = classificationAA.detach().cpu()
                        #Phone = classificationBB.detach().cpu()        
                
                
                        #classification_ori = torch.cat([ Human, Phone, ], 2)
                        ## (1, 15000, 2)
                        ####################################################################################################
                        #human = (classification_ori[:, :, 0].unsqueeze(2))#.detach().cpu().numpy() #.unsqueeze(2)
                        #phone = (classification_ori[:, :, 1].unsqueeze(2)) #.detach().cpu().numpy()
                
            
                        #################### EMPTY 한 애들 넣기 (442260, 2) => (442260, 90)
                        #class1 = torch.empty([8, 442260, 75]).detach().cpu() 
                        #class2 = torch.empty([8, 442260, 13]).detach().cpu() 
                
                
                        ####### 
                        #Human = human.detach().cpu()
                        #Phone = phone.detach().cpu()
                
                
                
                       
                        #classification = torch.cat([ Human, class1, Phone, class2], 2).cuda()
                        
                        
                        out = postprocess(concatimagesfinal, anchors, regression, classification, RegressB, ClipB, threshold=0.01,  iou_threshold=0.1)
                        
                        
                        prediction2 = invert_affine(framed_metas, out)
             
                        
                
            
                        INDEXs = np.array(INDEXs)
                        
                        ROIS_SET = []
                        CLASS_IDs = []
                
            
                        
                        for jjjj in range(classification.shape[0]):     
                            #
                            # Provided LABELS
                            #
                            score = SCOREs[jjjj]
                            index = INDEXs[jjjj]
                            rois = ROISs[jjjj]
                            classIDgt = CLASSIDs[jjjj]
                            
                            score = np.array(np.load(score))
                            classIDgt = np.array(np.load(classIDgt))
                            index = np.array(np.load(index))
                            rois = np.array(np.load(rois))
                            
                            
                            
                            
                            
                            CLASS_IDs.append(classIDgt)
                            ROIS_SET.append(rois)
                      
                        CLASS_IDs = np.array(CLASS_IDs, dtype = 'object') 
                        ROIS_SETs = np.array(ROIS_SET, dtype = 'object')
                        
                        
                        classification_predicted = torch.empty(concatimagesfinal.shape[0], 200, 2)
                        classification_true = torch.empty(concatimagesfinal.shape[0], 200, 2)    
                        
                        #
                        # PER BATCH
                        #
                        lengths = []
                        for jjjj in range(classification.shape[0]): 
                            
                            
                            
                            pROIS = prediction2[jjjj]["rois"]
                            pCLASSIDs = prediction2[jjjj]["class_ids"]
                            #pSCOREs = prediction2[jjjj]["scores"]
                            pSCOREpersons = prediction2[jjjj]["scores_person"]
                            pSCOREphones = prediction2[jjjj]["scores_phone"]
                            
                            lengths.append(len(pCLASSIDs))
                        #prediction = load_prediction(json_path="/home/glen/EfficientDet/predictionBATCH.json")
                            ROIS_SET_batch = ROIS_SETs[jjjj]
                            CLASS_ID_batch = CLASS_IDs[jjjj]
                            cla_true = np.zeros((200,2))
                            
                            
                            cla_pred = np.empty((1,2))
                            box_gt = np.empty((1,4))
                            box_gts = []
                            
                            #for i in range(len(pCLASSIDs)):
                            #    if(pCLASSIDs[i] != 0 and  pCLASSIDs[i] != 76):
                            #        pCLASSIDs[i] = 0
                            #        pSCOREs[i] = 0
                        
                        
                        #maximumscore = 0
                            for i in range(len(CLASS_ID_batch)): #and SCORE_batch[i] > maximumscore
                                if(CLASS_ID_batch[i] == 76):
                
                #maximumscore = SCORE_batch[i]
                                    box_gt = np.reshape(ROIS_SET_batch[i], [1,4])
                                    box_gts.append(box_gt)
        
                            box_gts = np.array(box_gts)      
                            box_gts = np.reshape(box_gts, [np.shape(box_gts)[0], 4])
        
                            
                            
                            
                             
                           
                            MSEs = []
                            ious = []
                            TRUE = box_gts
                            
                            ## 원래는
                            box_preds = pROIS
                            
                            for i in range(len(pCLASSIDs)):
                
                
                                #원래는 prediction_i, MSE, iou, mSEs, ious가 여기 있었음
                                prediction_i = np.reshape(np.array([box_preds[i,:]]), [1,4])
                                
            
                                
                                MSE = np.sqrt(np.sum(np.square(prediction_i - TRUE)))
                                iou = bbox_iou(prediction_i, TRUE)
                                MSEs = np.append(MSEs, MSE)
                                ious = np.append(ious, iou)
                
                                
                                AA = np.array( [ float((pSCOREpersons[i]*10000)/10000),  float((pSCOREphones[i]*10000)/10000)  ] )
                                AA = np.reshape(AA, [1,2])
                    
                                cla_pred = np.concatenate([cla_pred, AA], axis = 0)
                    
                                
                            cla_pred = cla_pred[1:]
                            
                            
                            ##############추가된 부분############################
                            #classification_original_scores = np.max(cla_pred, 1)
                
                            #class_scores_indices = np.argsort(classification_original_scores)[::-1]
                            #cla_pred = cla_pred[class_scores_indices, :]
            
                
                            #pROIS = pROIS[class_scores_indices, :]
                            
                            #####################원래 있었던 부분: ########################   
                            #print(np.shape(cla_pred))
                            if len(cla_pred) <= 200:
                                cla_pred = np.concatenate([cla_pred, np.zeros((200-len(cla_pred),2))], axis = 0 )
                    
                            else:
                                cla_pred = cla_pred[:200]
                
                            ############추가된 부분 2 #############################
                            #clascoreoriginal = cla_pred
                            
                            #clascoreoriginal = np.array(clascoreoriginal)
                
                
            
                            #classification_a = np.argmax(clascoreoriginal, 1)
                           
                
                
                
                            #
                            # phone인 class score와 confidence score 앞으로 옮기기
                            #
                            #indicesofphone = []
                            #for i in range(classification_a.shape[0]):
                   
                    
                            #    if classification_a[i] == 1:
                            #        indicesofphone.append(i)
                
                            #clascoreoriginal_phones = clascoreoriginal[indicesofphone, :]
                            #clascoreoriginal_persons = np.delete(clascoreoriginal, indicesofphone, 0)
                            #clascoresorted = np.concatenate([clascoreoriginal_phones, clascoreoriginal_persons] , 0)
                
                            #ROIS_phones = pROIS[indicesofphone, :]
                            #ROIS_persons = np.delete(pROIS, indicesofphone, 0)
                            #ROISsorted = np.concatenate([ROIS_phones, ROIS_persons], 0)
                
                            #box_preds = ROISsorted
                            #for i in range(len(box_preds)):
                            
                
                
                                
                            #    prediction_i = np.reshape(np.array([box_preds[i,:]]), [1,4])
                                
            
                                
                            #    MSE = np.sqrt(np.sum(np.square(prediction_i - TRUE)))
                            #    iou = bbox_iou(prediction_i, TRUE)
                            #    MSEs = np.append(MSEs, MSE)
                            #    ious = np.append(ious, iou)
                
                            #cla_pred = clascoresorted
                            #####################################################
                            #print('-----ACTUAL----')
                            #print(cla_pred)
                            
                            
                            #
                            # 모델의 INPUT
                            #
                            classification_predicted[jjjj, :, :] = torch.Tensor(cla_pred)
                            
                            
                            ious = np.array(ious)
                            MSEs = np.array(MSEs)
                            
                            #print(np.argsort(ious)[::-1])
                            
                            selectedboxesindices = []
                        
                            for i in range(len(ious)):
                                if(ious[i] > 0):
                                    
                                    selectedboxesindices.append(i)
                                
                                    
                            selectedboxesindices = np.array(selectedboxesindices)
                            
                            #argmin_index = np.where(np.array(MSEs) == minimum)
                            #argmin_index = int(argmin_index[0])
                            
                            #
                            # MAKE TRUE LABELS
                            #
                            #count1 = 0
                            #count2 = 0
                            
                            for i in range(len(pCLASSIDs)):
                                #################################################
                                # SET the boxes having IoU greater than threshold to phone box 
                                # Else to other box
                                #################################################
                                #if (i == argmin_index):
                                if((i in selectedboxesindices) == True):
                                    cla_true[i, 1] = 1
                                    #count1 += 1
                                    
                                else:
                                    cla_true[i, 0] = 1
                                    #count2 += 1
                            #print('-------------------------------------------')
                            #print(count1)
                            #print(count2)
                            #print('-------------------------------------------')
                            #
                            # 진짜 라벨이 되어야 할 것
                            #
                            classification_true[jjjj, :, :] = torch.Tensor(cla_true)  
                            
                            
                   
                    # classification_f: [100, 2]
                    
                    cls_loss1 = fus_model2(framed_metas, classification_predicted, classification_true, lengths)
                    
                    cls_loss1_tot += cls_loss1
                    
                    #reg_loss_tot += reg_loss
                    ########################################################
                    #cls_loss, reg_loss = fus_model(regression, regression_depth, classification, classification_depth, anchors, anchors_depth, annot)
                    
                    
                   
                    #cls_loss1_avg = cls_loss1_tot/len(framed_img1) 
                    #cls_loss2_avg = cls_loss2_tot/len(framed_img1)
                    #reg_loss_avg = reg_loss_tot/len(framed_img1)
                    #fro_loss1_avg = fro_loss1_tot*1000 #/len(framed_img1)
                    #fro_loss2_avg = fro_loss2_tot*1000 #/len(framed_img1)
                    #cls_loss_avg = cls_loss_tot.mean()
                    #reg_loss_avg = reg_loss_tot.mean()
                    
                    
                    
                    loss = cls_loss1_tot 
                    
                    #previous_loss = 0
                    
                    
                    
                    #if torch.isnan(loss):
                        
                    #    loss = previous_loss
                        
                    #    progress_bar.set_description(
                    #    'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss1: {:.5f}.  Total loss: {:.5f}'.format(
                    #        step, (epoch), (opt.num_epochs + 30), (iter + 1), num_iter_per_epoch, cls_loss1_tot.item(), 
                    #        loss))
                        
                    #    step += 1
                    #else:
                    if loss == 0 or not torch.isfinite(loss) :
                        
                        
                        continue
                    
                    
                    loss.requires_grad_(True)
                    loss.backward()
                    
                    
                    optimizer1.step()
                    
                    
                    #optimizer2.step()
                    
                    
                    epoch_loss.append(float(loss))
                    #print('Step: ' + str( step))
                    #print('Epoch: ' + str(epoch))
                    #print('Iteration: ' + str(iter + 1))
                    #print('Cls loss1: ' + str(cls_loss1_avg))
                    #print('Cls loss2: ' + str(cls_loss2_avg))
                    #print('Step: ' + str(step) + ' Epoch: ' + str(epoch) + ' Iteration: ' + str(iter + 1) +  ' Total loss: ' + str(loss))
                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss1: {:.5f}.   Total loss: {:.5f}'.format(
                            step, (epoch), (opt.num_epochs), iter + 1, num_iter_per_epoch, cls_loss1_tot.item(), 
                            loss.item()))
                    #print('Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss1: {:.5f}. fro loss1: {:.5f}. Total loss: {:.5f}'.format(
                    #        step, (epoch), (opt.num_epochs), iter + 1, num_iter_per_epoch, cls_loss1_avg.item(), cls_loss2_avg.item(), 
                    #        loss.item()))
                    # opt.num_epochs + 60
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Cls_loss 1', {'train': cls_loss1_tot}, step)
                        

                    # log learning_rate
                    current_lr = optimizer1.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1
                    
                    if step % opt.save_interval == 0 and step > 0:
                       
                        
                        save_checkpoint(model2, f'FTFCN-d{opt.compound_coef}_{opt.lr}_{epoch}_{step}_BATCH32.pth')
                        
              
                    
                        
                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler1.step(np.mean(epoch_loss))
            #scheduler2.step(np.mean(epoch_loss))
            train_losses.append(epoch_loss)
            #np.save('TRAIN_LOSSES.npy', train_losses)
            #if epoch % opt.val_interval == 0:
            save_checkpoint(model2, f'FTFCN-d{opt.compound_coef}_{opt.lr}_{epoch}_{step}_BATCH32.pth')  
            ########################## VALIDATION #######################################
            #print('VALIDATION')     
                
                
            #f1score = validate( model2)
                
                        # remember best f1 score 
            #is_best = f1score > best_f1score
            #best_f1score = max(f1score, best_f1score)
                
                        # save checkpoint depending on best_f1score
            #if is_best == True:
                    
                    
            #    save_checkpoint(model2, f'MMFWDL_MixerFinal-d{opt.compound_coef}_{opt.lr}_{epoch}_{step}_BEST_BATCH32.pth')
                
            #    print('Saved BEST Mixer!')
            #else:
                    
            #    print('Checkpoint NOT CHANGED')
                    
                    
                
            
                
    except KeyboardInterrupt:
        
        save_checkpoint(model2,f'FTFCN-d{opt.compound_coef}_{opt.lr}_{epoch}_{step}_BATCH32.pth')
        
        #a = torch.Tensor([weightA, weightB])
        #b = a.numpy()
        #np.save('/home/glen/VisionWireless/coefficients', b)
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    #if isinstance(model, CustomDataParallel):
    #    torch.save(model.module.state_dict(), os.path.join(opt.saved_path, name))
    #else:
    torch.save(model.state_dict(), os.path.join(opt.saved_path, name), _use_new_zipfile_serialization=False)

def save_checkpoint2(model, name):
    torch.save(model.module.state_dict(), os.path.join(opt.saved_path, name), _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    opt = get_args()
    train(opt)