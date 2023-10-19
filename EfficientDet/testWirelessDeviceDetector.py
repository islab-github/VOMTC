# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:34:25 2023


"""
import torch.nn.functional as F
import json
from collections import OrderedDict
import copy
from FT_FCN import build_DNN_refiner
#from load_test import load_prediction, load_data
from VOMTC_dataset_forVALTEST import load_test_data
### IF VOTRE:
#from VOTRE_dataset import load_val_data

def load_prediction(json_path: str = "data/prediction.json"):
    with open(json_path, "r") as f:
        pred = json.load(f)

    pred = OrderedDict(sorted(pred.items()))
    print(f"Prediction loaded, total: {len(pred)} predictions")
    return pred

from parse import parse_distance, mean_distance, center_angle

import numpy as np
import torch
import cv2
from net.backbone import EfficientDetBackbone
from net.efficientdet.utils import BBoxTransform, ClipBoxes
from net.utils.utils import plot_one_box, get_index_label
from utils import match_bboxes, non_max_suppression_fast, obj_list, color_list

#
# TEST data에 따라 다름
#

#from load_test import load_data


#from load_multi import load_data_partial
import math
from typing import Union

MODEL_TYPE = 8
RegressB = BBoxTransform()
ClipB = ClipBoxes()
PredJson = "predictionWirelessDeviceDetector_TEST_FULL.json"

PERSON_ID = 0
PHONE_ID = 76
PAD_RATIO = 0.3
from torchvision.ops.boxes import batched_nms

import xml.etree.ElementTree as ElemTree

def findcenter2D(bbox):
    
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    h, w = dist.shape
    x1 = min(w, max(x1, 0))
    x2 = min(w, max(x2, 0))
    y1 = min(h, max(y1 , 0))
    y2 = min(h, max(y2, 0))
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    return center_x, center_y
    
def parse_xml(xml_path: str, target: str = "phone"):
    assert target in ("nonphone", "phone")
    tree = ElemTree.parse(xml_path)
    objects = tree.findall("object")
    name = [x.findtext("name") for x in objects]

    result = []
    for i, object_iter in enumerate(objects):
        if ("P" in name[i]) and (target == "phone"):  # phone
            box = object_iter.find("bndbox")  # noqa
            result.append([int(it.text) for it in box])  # (x1, y1, x2, y2)
            
            
            
        elif ("P" not in name[i]) and (target == "nonphone"):  # notphone
            box = object_iter.find("bndbox")  # noqa
            result.append([int(it.text) for it in box])  # (x1, y1, x2, y2)
            
            
    return result


def parse_prediction(pred: dict, target: str = "phone", threshold: float = 0.0):
    assert target in ("nonphone", "phone")
    result = []
    for v in pred.values():
        if v["score"] < threshold * 100:
            continue

        if ("cell phone" in v["obj"]) and (target == "phone"):
            box = [v["x1"], v["y1"], v["x2"], v["y2"]]
            result.append(box)
        elif ("nonphone" in v["obj"]) and (target == "nonphone"):
            box = [v["x1"], v["y1"], v["x2"], v["y2"]]
            result.append(box)
    return result



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
        #(90, 442260)
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

def aspectaware_resize_padding_ORI(image, width, height, interpolation=None, means=None):
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

def preprocess(image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    #ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    ori_imgs = cv2.imread(image_path)
    
    # ori_imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in image_path]
    
    
    #각 이미지마다 normalize
    normalized_imgs = (ori_imgs[..., ::-1] / 255 - mean) / std 
    
    #각 normalized 이미지마다 aspectaware resize padding
    imgs_meta = [aspectaware_resize_padding_ORI(normalized_imgs, max_size, max_size,
                                            means=None)]
    
    #각 imgs_meta 이미지마다 0번째 index에 있는 값은 framed_img
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    
    #각 img_meta 이미지마다 나머지 index에 있는 값들은 framed_metas
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas
#
def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                #new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds

def load_model(model_type: int = MODEL_TYPE):
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    model = EfficientDetBackbone(
        compound_coef=MODEL_TYPE, num_classes=90,
        anchor_ratios=anchor_ratios, anchor_scales=anchor_scales
    )
    model.load_state_dict(torch.load(f"logs/coco/efficientdet-d{model_type}.pth", map_location="cpu"), strict=True)
    model = model.cuda()
    model.eval()
    return model, input_sizes[model_type]


    

def detect(framed_metas, model, img, threshold: float, iou_threshold: float):
    features, regression, classification, anchors = model(img)

    out = postprocess(
        img,
        anchors, regression, classification,
        RegressB, ClipB,
        threshold, iou_threshold
    )
    out = invert_affine(framed_metas, out)
    return out


def display(pred, img, img_key: str, display_objects=("person", "cell phone"), save: bool = False):
    img = img.copy()
    for j in range(len(pred['rois'])):
        x1, y1, x2, y2 = pred['rois'][j].astype(np.int32)
        obj = obj_list[pred['class_ids'][j]]
        if obj in display_objects:
            score = float(pred['scores'][j])
            if score > 0.3:
                plot_one_box(img, [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

            

    if save:
        cv2.imwrite(f"testResult/{img_key}", img)


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




model2 = build_DNN_refiner()

state_dict_path2 = '/home/glen/EfficientDet/logs/VisionWireless/BEST_BATCH32.pth'

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:2")
device3 = torch.device("cuda:3")
input_size = 1536 #state_dict_path1, state_dict_path2
def test(model2):
    model, input_size = load_model(model_type=MODEL_TYPE)

    loaded_state_dict2 = torch.load(state_dict_path2)
    new_state_dict2 = OrderedDict()
    for n, v in loaded_state_dict2.items():
        name = n.replace("model1.","") 
        new_state_dict2[name] = v

    model2.load_state_dict(new_state_dict2)
    
    print('MODELS LOADED 0 207')
    
    model2.eval().to(device2)
    
    

    torch.set_grad_enabled(False)

    json_prediction = OrderedDict()
    
    data = load_test_data() 
    total = len(data)
   
    #for count, (data_key, (gt_path, rgb_path, depth_path, distance_path, directory, original_file)) in enumerate(data.items()):
    for count, (data_key, (gt_path, rgb_path, distance_path, directory, original_file)) in enumerate(data.items()):
        
        img_key = rgb_path.split('/')[-1]
        
        # NECESSARY
        #depth_path = depth_path.replace(".png", "_depth_color_image.png")
        ####
        print(f"{count} / {total}, {img_key}")
        
        print(rgb_path)
       
        ori_img, framed_img, framed_metas = preprocess(rgb_path, max_size=input_size)

        img_th = torch.from_numpy(framed_img[0]).cuda().unsqueeze(0).permute(0, 3, 1, 2)
        prediction = detect(
            framed_metas, model, img_th,
            threshold=0.4,
            iou_threshold=0.4
        )[0]

        crop_prediction = copy.deepcopy(prediction)  # dict
        for j in range(len(prediction['rois'])):
            obj = obj_list[prediction['class_ids'][j]]
            if obj == "person":
                x1, y1, x2, y2 = prediction['rois'][j].astype(np.int32)
                crop_img, (crop_x1, crop_y1, crop_x2, crop_y2) = crop(ori_img, (x1, y1, x2, y2), pad_ratio=PAD_RATIO)
                crop_path = f"crop_RGB/{img_key[:-4] + 'crop' + str(j) + '.png'}"
                if crop_img.size == 0:
                    continue

                cv2.imwrite(crop_path, crop_img)
                #cropped image is the input
                _, framed_img, framed_metas = preprocess(crop_path, max_size=input_size)
                img_th = torch.from_numpy(framed_img[0]).cuda().unsqueeze(0).permute(0, 3, 1, 2)
                
                
                prediction2 = detect(
                    framed_metas, model, img_th,
                    threshold=0.01,
                    iou_threshold=0.1
                )[0]
                ######################
                
                CLASSIDs = prediction2["class_ids"]
                
                #SCOREs = prediction2["scores"]
                
                ROISs = prediction2["rois"]
                
                SCOREpersons = prediction2['scores_person']
                SCOREphones = prediction2['scores_phone']
                
                cla_pred = np.empty((1,2))
                
                            
                for i in range(len(CLASSIDs)):
            
                
                
                
                                
                    AA = np.array( [ float((SCOREpersons[i]*10000)/10000),  float((SCOREphones[i]*10000)/10000)  ] )
                    AA = np.reshape(AA, [1,2])
                    
                    cla_pred = np.concatenate([cla_pred, AA], axis = 0)
                       
                cla_pred = cla_pred[1:]
                            
                    
                #print(len(cla_pred))
                ##########################################
                classification_original_scores = np.max(cla_pred, 1)
                
                class_scores_indices = np.argsort(classification_original_scores)[::-1]
                cla_pred = cla_pred[class_scores_indices, :]
            
                
                ROISs = ROISs[class_scores_indices, :]
                
                
                #############################################            
                #
                # 여기서 학습한 모델이 cla_pred에서 cla_true로 바꾸는 역할해줘야함!
                #classification_f = cla_pred
                #
                
                if len(cla_pred) <= 200:
                    cla_pred = np.concatenate([cla_pred, np.zeros((200-len(cla_pred),2))], axis = 0 )
                    
                else:
                    cla_pred = cla_pred[:200]
                    
                clascoreoriginal = cla_pred
                #clascoreoriginal = clascoreoriginal[:len(CLASSIDs)]
                clascoreoriginal = np.array(clascoreoriginal)
                
                
            
                classification_a = np.argmax(clascoreoriginal, 1)
                
                
                
                
                #
                # phone인 class score와 confidence score 앞으로 옮기기
                #
                indicesofphone = []
                for i in range(classification_a.shape[0]):
                   
                    
                    if classification_a[i] == 1:
                        indicesofphone.append(i)
                
                clascoreoriginal_phones = clascoreoriginal[indicesofphone, :]
                clascoreoriginal_persons = np.delete(clascoreoriginal, indicesofphone, 0)
                clascoresorted = np.concatenate([clascoreoriginal_phones, clascoreoriginal_persons] , 0)
                
                ROIS_phones = ROISs[indicesofphone, :]
                ROIS_persons = np.delete(ROISs, indicesofphone, 0)
                ROISsorted = np.concatenate([ROIS_phones, ROIS_persons], 0)
                
                
                
                cla_pred = clascoresorted
                
                
                classification_scores = np.max(cla_pred[:len(CLASSIDs)], 1)
                
                
                
                ############################
                
                cla_pred = np.reshape(cla_pred, [1, np.shape(cla_pred)[0]* np.shape(cla_pred)[1] ])
                classification_resized = torch.Tensor(cla_pred)
                classification_resized = classification_resized.detach().cpu()
        
        
        
                classification_input = classification_resized
        
        
        
       
                classification_output = model2(classification_input.to(device2))
        
        
        
                classification_output = torch.reshape(classification_output, [classification_output.shape[0], int(classification_output.shape[1]/2), 2]).detach().cpu()
                
                
                
                classification_f = torch.Tensor(classification_output)
                            
                
                            
                classification_f = classification_f.squeeze(0)
                classification_f = F.softmax(classification_f)
                
                cla_pred = classification_f
                ###################################
                cla_pred = cla_pred[:len(CLASSIDs)]
                cla_pred = np.array(cla_pred)
                
                
                #classification_scores = np.max(cla_pred , 1)
                classification_f = np.argmax(cla_pred, 1)
                
                for i in range(classification_f.shape[0]):
                   
                    ## ASSIGN CELL PHONE
                    if classification_f[i] == 1:
                       
                        classification_f[i] = 76
                    
                    ## ASSIGN NON CELL PHONE
                    elif classification_f[i] == 0:
                      
                        classification_f[i] = 90
                        
                    
                        
                      
                
                
                if len(ROISsorted) > 0:
                    ROISsorted[:, 0] += crop_x1
                    ROISsorted[:, 1] += crop_y1
                    ROISsorted[:, 2] += crop_x1
                    ROISsorted[:, 3] += crop_y1
                    
                    crop_prediction["rois"] = np.concatenate((crop_prediction["rois"], ROISsorted))
                    crop_prediction["class_ids"] = np.concatenate(
                        (crop_prediction["class_ids"], classification_f))
                    crop_prediction["scores"] = np.concatenate((crop_prediction["scores"], classification_scores))
               

        if data_key in json_prediction.keys():
            raise ValueError(f"data key error, {data_key}")
        json_prediction[data_key] = OrderedDict()
        json_count = 0
        
        
        
        if len(crop_prediction['rois']) > len(crop_prediction['class_ids']):
            print('1')
            crop_prediction['rois'] =  crop_prediction['rois'][:len(crop_prediction['class_ids'])]
            
        elif len(crop_prediction['rois']) < len(crop_prediction['class_ids']):
            print('2')
            crop_prediction['class_ids'] =  crop_prediction['class_ids'][:len(crop_prediction['rois'])]
        
        display(crop_prediction, ori_img, img_key, display_objects=("person", "cell phone"), save=True)
        
        for j in range(len(crop_prediction['rois'])):
            obj = obj_list[crop_prediction['class_ids'][j]]
            
            if obj != "cell phone":
                x1, y1, x2, y2 = crop_prediction['rois'][j].astype(np.int32)
                json_prediction[data_key][str(json_count)] = OrderedDict(
                    obj="nonphone", x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    score=int(crop_prediction['scores'][j] * 100))
                json_count += 1
            elif obj == "cell phone":
                x1, y1, x2, y2 = crop_prediction['rois'][j].astype(np.int32)
                json_prediction[data_key][str(json_count)] = OrderedDict(
                    obj="cell phone", x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    score=int(crop_prediction['scores'][j] * 100))
                json_count += 1
            
        # if count == 10:
        #     break
    
    
    with open(PredJson, "w") as f:
         json.dump(json_prediction, f)
                
TARGET = "phone"  # nonphone


if __name__ == '__main__':
    test(model2)


    data = load_test_data()
    predictionF = load_prediction(json_path="/home/glen/EfficientDet/predictionWirelessDeviceDetector_TEST_FULL.json")
    angle = np.load("degree.npy")
    assert len(data) == len(predictionF), f"Not matched {len(data)} vs {len(predictionF)}"
    total = len(data)

    tp = 0
    fp = 0
    fn = 0
    dist_err = 0.0
    dist_valid_count = 0

    
    angle_err = 0.0
    angle_valid_count_2 = 0
    
    azimuth_angle_err = 0
    elevation_angle_err = 0
    angle_valid_count = 0
    
    omitted_samples = 0
    failed_samples = 0
    total_samples = 0
    
    omitted_samples = 0
    failed_samples = 0
    total_samples = 0
    imagewhereprecisiondegrades = []
    imagewhere_recalldegrades = []
    values = []
    
    #az_sample_num = 0
    #el_sample_num = 0

    # already sorted
    for count, ((gt_path, rgb_path, distance_path, _,_), pred) in enumerate(zip(data.values(), predictionF.values())):
        print(f"{count} / {total}, {rgb_path}")

        #if ("v2_cal1_" in gt_path) or ("cal211_" in gt_path) or ("cal241_" in gt_path)\
        #        or ("cal121_" in gt_path) or ("cal61_" in gt_path):
        #    omitted_samples += 1
        #    continue

        dist = parse_distance(distance_path)
        box_gt = np.array(parse_xml(gt_path, target=TARGET))
        box_pred = np.array(parse_prediction(pred, target=TARGET, threshold=0.1))
        
        box_pred = non_max_suppression_fast(
                box_pred,
        iou_threshold=0.1 if (TARGET == "phone") else 0.5
        )
        
        idx_gt, idx_pred, iou_score = match_bboxes(
                box_gt, box_pred,
                threshold=0.01 if (TARGET == "phone") else 0.1
        )
        
        num_gt = len(box_gt)
        num_pred = len(box_pred)
        num_match = len(idx_gt)
    
        print(f"... GT: {num_gt} / Pred: {num_pred} / Match: {num_match}, score: {iou_score}")

        #if (num_gt > 5) or (num_match == 0):
        #    failed_samples += 1
        #    total_samples += 1
            
        #    continue
        total_samples += num_gt
        tp += num_match
        fn += (num_gt - num_match)
        fp += (num_pred - num_match)

        for ig, ip in zip(idx_gt, idx_pred):
            angle_valid_count += 1
            bg = box_gt[ig]
            bp = box_pred[ip]
            
            r_gt =  mean_distance(dist, bg)
            r_predicted = mean_distance(dist, bp)
            
            #dist_err = abs(r_gt - r_predicted)
            
            ## in degrees
            azimuth_gt = center_angle(angle, bg)
            azimuth_predicted = center_angle(angle, bp)
            
            x_gt, y_gt = findcenter2D(bg)
            x_predicted, y_predicted = findcenter2D(bp)
            elevation_gt = (math.atan(y_gt/x_gt))* (180/math.pi)
            elevation_predicted = (math.atan(y_predicted/x_predicted))* (180/math.pi)
            
            #if (elevation_gt > 0) and (elevation_predicted > 0):
            elevation_angle_err += abs(elevation_gt - elevation_predicted)
            #    el_sample_num += 1
                
            #if (azimuth_gt > 0) and (azimuth_predicted > 0):
            azimuth_angle_err += abs(azimuth_gt - azimuth_predicted) 
            #    az_sample_num += 1
            
            
            
    
    failed_sample_num = total_samples - tp
    elevation_angle_err += (total_samples - tp)*abs(int(np.max(angle)) - 1)
    azimuth_angle_err += (total_samples - tp)*abs(int(np.max(angle)) - 1)  
            
    precision = (tp / (tp + fp))*100
    recall = (tp / (tp + fn))*100
    mean_azimuth_angle_err = azimuth_angle_err / total_samples #(az_sample_num + failed_sample_num)  #
    mean_elevation_angle_err =  elevation_angle_err / total_samples #(el_sample_num + failed_sample_num)  #

    f1_score = 2*((precision*recall)/(precision + recall) )
    
    #ratio = (total_samples - failed_samples)/(total_samples)
    #real_f1_score = f1_score*(ratio)
    print(f"Precision: {precision:.6f} (TP: {tp}, FP: {fp})")
    print(f"Recall: {recall:.6f} (TP: {tp}, FN: {fn})")
    print(f"Azimuth Angle error: {mean_azimuth_angle_err:6f} deg (valid: {angle_valid_count})")
    print(f"Elevation Angle error: {mean_elevation_angle_err:6f} deg (valid: {angle_valid_count})")
    #print(f"Distance error: {mean_dist_err * 100:6f} cm (valid: {dist_valid_count})")
    #print(f"Angle error: {mean_angle_err:6f} deg (valid: {angle_valid_count_2})")
    #print(f"Omitted samples: {omitted_samples}")        
    #print(f"Failed samples: {failed_samples}")
    print(f"F1 SCORE: {f1_score}")
    print(f"total samples: {total_samples}")
    #print(f"F1 SCORE times failed ratio: {real_f1_sco
    
       
    #print(f"F1 SCORE times failed ratio: {real_f1_score}")


  
        
        
    
    

  