# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:50:36 2022

@author: Glen Kim
"""


from collections import OrderedDict

from VOMTCdatasetSelection import VOMTCdatasetSelection
#from VOTREdatasetGeneration import VOTREdatasetGeneration


import numpy as np
import torch
import cv2
from net.backbone import EfficientDetBackbone
from net.efficientdet.utils import BBoxTransform, ClipBoxes
from net.utils.utils import preprocess_UPDATED, invert_affine, plot_one_box, get_index_label
import xml.etree.ElementTree as ElemTree
from utils import obj_list, color_list
from parse import parse_distance

###############################################################################
#
# Run the VOTREdatasetSelection code to obtain the desired dataset consisting of selected pairs of RGB and depth images 
# satisfying the three user-specified parameters.
# 
###############################################################################
activeClasses = [0, 1]
maxnumPeople = 6
maxDist = 30 
data = VOMTCdatasetSelection(activeClasses, maxnumPeople, maxDist)
#data = VOTREdatasetGeneration(activeClasses, maxnumPeople, maxDist)
print('Dataset Selected!')
total = len(data)
print('Selected Data SIZE')
print(total)

################################################################################
MODEL_TYPE = 8
RegressB = BBoxTransform()
ClipB = ClipBoxes()

PERSON_ID = 0
PHONE_ID = 76
LAPTOP_ID = 72
NONE_ID = 90
PAD_RATIO = 0.3



def Mask(img, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img[y1:y2, x1:x2, :] = 0
    masked_img = img 
    return masked_img

def HumanMask(img, cropped_mobile_img, box):
    
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    img[:, :, :] = 0 
    img[y1:y2, x1:x2, : ] = cropped_mobile_img
    masked_img = img 
    return masked_img

def MaskDepth(img, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img[y1:y2, x1:x2] = 0
    masked_img = img 
    return masked_img

def HumanMaskDepth(img, cropped_mobile_img, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    img[:, :] = 0 
    img[y1:y2, x1:x2] = cropped_mobile_img
    masked_img = img 
    return masked_img

def cropHumanAndMobile(img, crop_img, mobile_box, pad_ratio: float = PAD_RATIO):
    img_h, img_w = img.shape[:2]
    box = crop_img
    ###### human ########
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = (x2 - x1), (y2 - y1)
    pad_w = (w * pad_ratio) // 2
    pad_h = (h * pad_ratio) // 2
    pad = max(pad_w, pad_h)
    new_x1, new_y1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    new_x2, new_y2 = int(min(x2 + pad, img_w)), int(min(y2 + pad, img_h))
    ###### phone ###########
    x1_mobile, y1_mobile, x2_mobile, y2_mobile = int(mobile_box[0]), int(mobile_box[1]), int(mobile_box[2]), int(mobile_box[3])
    w_mobile, h_mobile = (x2_mobile - x1_mobile), (y2_mobile - y1_mobile)
    pad_w_mobile = (w_mobile * pad_ratio) // 2
    pad_h_mobile = (h_mobile * pad_ratio) // 2
    pad = max(pad_w_mobile, pad_h_mobile)
    new_x1_mobile, new_y1_mobile = int(max(x1_mobile - pad, 0)), int(max(y1_mobile - pad, 0))
    new_x2_mobile, new_y2_mobile = int(min(x2_mobile + pad, img_w)), int(min(y2_mobile + pad, img_h))
    ######### cropped Image #############
    
    crop_img = img[new_y1:new_y2, new_x1:new_x2, :].copy()
    
    ######### transformed coordinates of phone in cropped Image ########
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



device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:2")
device3 = torch.device("cuda:3")

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height,c), np.float32)
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




def preprocess(*image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    #ori_imgs = cv2.imread(image_path) 
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

    

def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold): #threshold:0.01
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
        
        #각 image마다 transformed anchor
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        
        #각 image마다 classification 점수들의 max인 점수와 해당 class
        scores_, classes_ = classification_per.max(dim=0)
        
        out.append({'rois': transformed_anchors_per.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
        })
    
        
    true_indices_set = np.array(true_indices_set) 
    true_indices_set = np.reshape(true_indices_set, [np.shape(true_indices_set)[2]])
    
    return out, true_indices_set
    

    
def two_detect_version2(framed_metas, img1, anchors, regression, classification, threshold:float, iou_threshold:float):
    
    out, true_indices_set  = postprocess(
        img1,
        anchors, regression, classification,
        RegressB, ClipB,
        threshold, iou_threshold
    )

    out = invert_affine(framed_metas, out)
    return out, true_indices_set

def load_model(model_type: int = MODEL_TYPE):
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    model = EfficientDetBackbone(
        compound_coef=MODEL_TYPE, num_classes=90,
        anchor_ratios=anchor_ratios, anchor_scales=anchor_scales
    )
    model.load_state_dict(torch.load(f"ckpt/efficientdet-d{model_type}.pth", map_location="cpu"), strict=True)
    model = model.cuda()
    model.eval()
    return model, input_sizes[model_type]
    



def display(pred, img, img_key: str, display_objects=("person", "cell phone"), save: bool = False):
    img = img.copy()
    for j in range(len(pred['rois'])):
        x1, y1, x2, y2 = pred['rois'][j].astype(np.int32)
        obj = obj_list[pred['class_ids'][j]]
        if obj in display_objects:
            score = float(pred['scores'][j])
            plot_one_box(img, [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

    if save:
        cv2.imwrite(f"test/{img_key}", img)


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





def run():
    model, input_size = load_model(model_type=MODEL_TYPE)
    #input_size = 1536
    torch.set_grad_enabled(False)

    json_prediction = OrderedDict()

    
    
    for count, (data_key, (gt_path, rgb_path1, distance_path, directory, file_name)) in enumerate(data.items()):
    
        img_key1 = rgb_path1.split('/')[-1]
        
        print(f"{count} / {total}, {img_key1}")
        
        ori_img1, framed_img1, framed_metas = preprocess_UPDATED(rgb_path1, max_size=input_size)
        
        
        ori_img2 = parse_distance(distance_path)
        
    
        
        if data_key in json_prediction.keys():
            raise ValueError(f"data key error, {data_key}")
        json_prediction[data_key] = OrderedDict()
        
        
        crop_prediction = {}
        
        
        crop_prediction["rois"] = np.zeros((1,4), dtype = np.int32)
        crop_prediction["class_ids"] = np.zeros((1) , dtype = np.int32)
        crop_prediction["scores"] = np.zeros((1) , dtype = np.int32)
        
        
        crop_paths = []
        
        
        
        
        tree = ElemTree.parse(gt_path)
        objects = tree.findall("object")
        name = [x.findtext("name") for x in objects]

        person_annotations = np.zeros((1,4))
        phone_annotations = np.zeros((1,4))
        
        
        for j, object_iter in enumerate(objects):
            
            if "person" in name[j]:
                box = object_iter.find("bndbox")  
                annotation = [int(it.text) for it in box]
                
                annotation = np.array(annotation).reshape(1,4)
                
                person_annotations = np.append(person_annotations, annotation, axis = 0)
                
            elif "P" in name[j]:
                box = object_iter.find("bndbox")  
                annotation = [int(it.text) for it in box]
                
                annotation = np.array(annotation).reshape(1,4)
                
                phone_annotations = np.append(phone_annotations, annotation, axis = 0)
                
        
        person_annotations = person_annotations[1:]
        phone_annotations = phone_annotations[1:]
        
        
        
        ALREADY_CHECKED_PERSON = []
        ALREADY_CHECKED_PHONE = []
        
        global classificationA
        global classificationB
       
        
        for ii in range(person_annotations.shape[0]):
            for jj in range(phone_annotations.shape[0]):
                
                #
                # IF PHONE is with HUMAN
                #
                
                if ((person_annotations[ii, 0]  <= phone_annotations[jj, 0] or person_annotations[ii, 1] <= phone_annotations[jj, 1])  and  (person_annotations[ii, 2] >= phone_annotations[jj, 2] or  person_annotations[ii, 3] >=  phone_annotations[jj, 3]) and ii not in ALREADY_CHECKED_PERSON and jj not in ALREADY_CHECKED_PHONE ):
                    ALREADY_CHECKED_PERSON.append(ii)
                    ALREADY_CHECKED_PHONE.append(jj)
                    
                    #######################################################################################################
                    #
                    # STEP 1: Obtain cropped RGB images (+ depth images) as inputs needed for training the VOTRE-based  
                    # object detector. cropHumanAndMobile includes simple coordinate transformation.
                    # 
                    #######################################################################################################
        
                    crop_img1, (crop_x1, crop_y1, crop_x2, crop_y2), crop_mobile_img1, (crop_x1_mobile, crop_y1_mobile, crop_x2_mobile, crop_y2_mobile) = cropHumanAndMobile(ori_img1[0], (person_annotations[ii, 0], person_annotations[ii, 1], person_annotations[ii, 2], person_annotations[ii, 3]), (phone_annotations[jj, 0], phone_annotations[jj, 1], phone_annotations[jj, 2], phone_annotations[jj, 3]), pad_ratio=PAD_RATIO)
                    crop_img2, _, crop_mobile_img2, _ = cropHumanAndMobile2D(ori_img2, (person_annotations[ii, 0], person_annotations[ii, 1], person_annotations[ii, 2], person_annotations[ii, 3]), (phone_annotations[jj, 0], phone_annotations[jj, 1], phone_annotations[jj, 2], phone_annotations[jj, 3]), pad_ratio=PAD_RATIO)
                     
                    crop_path1 = "C:/Users/Glen Kim/VisionWireless/cropped/RGB_cropped/"
                    crop_path1 = crop_path1 + f"{img_key1[:-4]}" + 'crop' + str(ii) + str(jj) + '.png'
                    
                    
                    
                    #crop_path2 = "C:/Users/Glen Kim/VisionWireless/cropped/depth_cropLabel/"
                    #crop_path2 = crop_path2 + f"{img_key1[:-4]}" + 'crop' + str(ii) + str(jj) + '.png'
                    print(crop_path1)
                    #print(crop_path2)
                    cv2.imwrite(crop_path1, crop_img1)
                    
                    #plt.imsave(crop_path2, crop_img2)
                    
                    
                    if crop_img1.size and crop_mobile_img1.size and crop_mobile_img2.size and crop_img2.size == 0:
                        continue
                    
                    _, framed_img_crop, framed_metas = preprocess_UPDATED(crop_path1, max_size = input_size)
                    #_, framed_depthimg_crop ,_ = preprocess(crop_path2, max_size = input_size)
                    

                    ori_img_th = torch.from_numpy(framed_img_crop[0]).cuda().unsqueeze(0).permute(0, 3, 1, 2)
                    #ori_depth_img_th = torch.from_numpy(framed_depthimg_crop[0]).cuda().unsqueeze(0).permute(0, 3, 1, 2)
                    
                    #######################################################################################
                    #
                    # STEP 2: We feed the cropped RGB images into the PT object detector to obtain 
                    # class score predictions and bounding box predictions for detected objects in each cropped image
                    #
                    #######################################################################################
                
                    features, regression, classification, anchors = model(ori_img_th)
                    ######################################################################################
                    #_, regressionDepth, classificationDepth, anchorsDepth = model(ori_depth_img_th)
                    
                    regressionF = regression
       
                    anchorsF = anchors
       
                    
                    classificationF = classification
                    
                    classificationA = classificationF[:, :, 0].unsqueeze(2)
                    classificationB = classificationF[:, :, 76].unsqueeze(2)
                    
                    
                #    
                # OTHERWISE, exclude the samples that do not contain cell phone.
                #
                
                elif (ii not in ALREADY_CHECKED_PERSON and jj not in ALREADY_CHECKED_PHONE and ii != jj):
                    ALREADY_CHECKED_PERSON.append(ii)
                    ALREADY_CHECKED_PHONE.append(jj)
                    crop_img1, (crop_x1, crop_y1, crop_x2, crop_y2) = crop(ori_img1[0], (person_annotations[ii, 0], person_annotations[ii, 1], person_annotations[ii, 2], person_annotations[ii, 3]), pad_ratio=PAD_RATIO)
                    
                    crop_path1 = f"crop_multi2/{img_key1[:-4] + 'crop' + str(ii) + str(jj)  + '.png'}"
                    
                    print(crop_path1)
                    if crop_img1.size == 0:
                        continue
                
                    
                    cv2.imwrite(crop_path1, crop_img1)
                
                    
                    crop_paths.append(crop_path1)
                    
                    _, framed_img1, framed_metas = preprocess_UPDATED(crop_path1, max_size = input_size)
                    
                    img_th1 = torch.from_numpy(framed_img1[0]).cuda().unsqueeze(0).permute(0, 3, 1, 2)
                
                    
                
                     
                    features1, regression1, classification1, anchors1 = model(img_th1)
                    

                
                    
                    classificationA = classification1[:, :, 0].unsqueeze(2)
                    classificationB = torch.empty([1, 442260, 1])
                    ori_img_th = img_th1  
                    
                
                    regression = regression1
                    anchors = anchors1
                    
                 
                class1 = torch.empty([1, 442260, 75]).detach().cpu() 
                class2 = torch.empty([1, 442260, 13]).detach().cpu() 
                
                Human = classificationA.detach().cpu()
                Phone = classificationB.detach().cpu()
                
                
                
                
                classification = torch.cat([ Human, class1, Phone, class2], 2) #.cuda()
                del class1
                torch.cuda.empty_cache()
                del class2
                torch.cuda.empty_cache()
                del Human
                torch.cuda.empty_cache()
                del Phone
                torch.cuda.empty_cache()
                
                #
                # postprocessing step 
                #
                
                prediction2, true_indices_set = two_detect_version2(framed_metas, ori_img_th, anchorsF.detach().cpu(), regressionF.detach().cpu(), classificationF, threshold=0.01, iou_threshold=0.1)
                
                prediction2 = prediction2[0]
                
                LENGTH = len(prediction2["class_ids"])
            
                setofPHONEindices = []
                MSEs = []
                
                
                #####################################################################
                #
                # Compute the mean squared error (MSE) measuring
                # the position error between each selected bounding box and the ground-truth bounding box,
                # and then identify the box with the smallest MSE error value.
                #
                #####################################################################
                TRUE = np.array([crop_x1_mobile, crop_y1_mobile, crop_x2_mobile, crop_y2_mobile])
                
                for j in range(LENGTH):
                    if(prediction2["class_ids"][j] == 76):
                        setofPHONEindices.append(j)
                 
                
                for i in range(LENGTH):
                    
                    MSE = np.sqrt(np.sum(np.square(prediction2["rois"][i,:] - TRUE)))
                    MSEs.append(MSE)
                MSEs = np.array(MSEs)
                
                
                minimum = min(MSEs)
                argmin_index = np.where(np.array(MSEs) == minimum)
                
                
                #####################################################################################################
                #
                # Just in case if all class predictions belong to person, set all class scores to 0 and none:
                # STEP 3: Using the bounding box of the cell phone obtained from simple coordinate transformation, 
                # we set the score and class of the identified box to 1 and cell phone, and save the ROIS of that box.
                #
                #####################################################################################################
                if all ([ v == 0 for v in prediction2["class_ids"]]):
                        
                    for z in range(LENGTH):
                        prediction2["scores"][z] = 0
                        prediction2["class_ids"][z] = NONE_ID
                        
                    prediction2["scores"][0] = 1
                    prediction2["class_ids"][0] = PHONE_ID
                    prediction2["rois"][0] =  np.array([crop_x1_mobile, crop_y1_mobile, crop_x2_mobile, crop_y2_mobile])
                    
                #########################################################################################################
                #
                # STEP 3: Using the bounding box of the cell phone obtained from simple coordinate transformation, 
                # we set the score and class of the identified box to 1 and cell phone, and save the ROIS of that box.
                # Otherwise, we set the score and class of the remaining boxes to 0 and none.
                #
                #########################################################################################################
                
                    
                else:     
                    
                
                    for z in range(LENGTH):
            
                        if z == argmin_index[0]:
                        
                            prediction2["scores"][argmin_index] = 1
                            prediction2["class_ids"][argmin_index] = PHONE_ID
                            
                            prediction2["rois"][argmin_index] = np.array([crop_x1_mobile, crop_y1_mobile, crop_x2_mobile, crop_y2_mobile])
                            
                        else:
                            prediction2["scores"][z] = 0
                            prediction2["class_ids"][z] = NONE_ID
                
                
                
                #
                # Among predicted boxes, excluding overlapping boxes and their confidence scores and class ids if there are any.
                #
                
                NONOVERLAPPING_ROIS = []
                NONOVERLAPPING_ROIS_indexes = []
                original_rois = prediction2["rois"]
                for i, num in enumerate(prediction2["rois"].tolist()):
            
                    if num not in NONOVERLAPPING_ROIS:
                        NONOVERLAPPING_ROIS.append(num)
                        NONOVERLAPPING_ROIS_indexes.append(i)
                        
                
                prediction2["rois"] = np.array(NONOVERLAPPING_ROIS)
                
                NONOVERLAPPING_CLASS_IDS = []
                NONOVERLAPPING_SCORES = []
                
                
                            
                for z in range(len(original_rois)):
                    if z in NONOVERLAPPING_ROIS_indexes:
                        NONOVERLAPPING_CLASS_IDS.append(prediction2["class_ids"][z])
                        NONOVERLAPPING_SCORES.append(prediction2["scores"][z])
                        
                prediction2["class_ids"] = np.array(NONOVERLAPPING_CLASS_IDS)
                prediction2["scores"] = np.array(NONOVERLAPPING_SCORES)
                
                CLASS_ID_VECTOR = prediction2["class_ids"]
                CONFIDENCE_VECTOR = prediction2["scores"]
                INDEXES = np.array(NONOVERLAPPING_ROIS_indexes)
                ROISs = prediction2["rois"] 
                
                ###############################################################################################################
                #
                # For cropped images containing both person and cell phone, save their class, gt-box, and scores in numpy files.
                # We use these as labels for training the VOTRE-based object detector.
                #
                ################################################################################################################
                
                substring = ('crop' + str(ii) + str(jj))
                if substring in crop_path1:
                    
                    path1 = 'C:/Users/Glen Kim/VisionWireless/cropped/dataLabel/CLASSID/'
                    path1 = path1 + f'{img_key1[:-4]}' + 'crop' + str(ii) + str(jj) +'.npy' 
                    np.save(path1 , CLASS_ID_VECTOR)
                    path2 = 'C:/Users/Glen Kim/VisionWireless/cropped/dataLabel/INDEX/'
                    path2 = path2 + f'{img_key1[:-4]}' + 'crop' + str(ii) + str(jj) +'.npy' 
                    np.save(path2, INDEXES)
                    path3 = 'C:/Users/Glen Kim/VisionWireless/cropped/dataLabel/GTBOX/'
                    path3 = path3 + f'{img_key1[:-4]}' + 'crop' + str(ii) + str(jj) +'.npy' 
                    np.save(path3 , ROISs)
                    path4 = 'C:/Users/Glen Kim/VisionWireless/cropped/dataLabel/SCORE/'
                    path4 = path4 + f'{img_key1[:-4]}' + 'crop' + str(ii) + str(jj) +'.npy' 
                    np.save(path4 , CONFIDENCE_VECTOR)
                    
                ##############################################################
                
                if len(prediction2["rois"]) > 0:
                    prediction2["rois"][:, 0] += crop_x1
                    prediction2["rois"][:, 1] += crop_y1
                    prediction2["rois"][:, 2] += crop_x1
                    prediction2["rois"][:, 3] += crop_y1
                    
                
                A = np.concatenate((crop_prediction["rois"], prediction2["rois"]))
                B = np.concatenate((crop_prediction["class_ids"], prediction2["class_ids"]))
                C = np.concatenate((crop_prediction["scores"], prediction2["scores"]))
                
                
                crop_prediction["rois"] = A[1:]
                crop_prediction["class_ids"] = B[1:]
                crop_prediction["scores"] = C[1:]
                
                del A
                del B
                del C
               
                LENGTHH = len(crop_prediction["class_ids"])
                
            
                #
                # Take only the predictions whose classes are 'cell phone'
                #
                reducedROIS = []
                reducedCLASSIDS = []
                reducedSCORES = []
                for z in range(LENGTHH):
                    if (crop_prediction["class_ids"][z] == 76):#원래는 76
                        reducedROIS.append(crop_prediction["rois"][z])
                        reducedCLASSIDS.append(crop_prediction["class_ids"][z])
                        reducedSCORES.append(crop_prediction["scores"][z])
                
                    
        
                reducedROIS = np.array(reducedROIS)
                reducedCLASSIDS = np.array(reducedCLASSIDS)
                reducedSCORES = np.array(reducedSCORES)
                                
                
                crop_prediction["rois"] = np.concatenate((reducedROIS , reducedROIS))
                crop_prediction["class_ids"] = np.concatenate((reducedCLASSIDS , reducedCLASSIDS))
                crop_prediction["scores"] = np.concatenate((reducedSCORES, reducedSCORES))
                
                print('---------------- CROP PREDICTION ROIS -----------------')
                print(crop_prediction["class_ids"])
                print(crop_prediction["scores"])
                print(crop_prediction["rois"])
                print(crop_prediction["class_ids"])
                print(len(crop_prediction["class_ids"]))
                print(crop_prediction["scores"])
                print(len(crop_prediction["scores"]))
                
             
        #
        # For each Full Image Sample
        # 
        #
        FreducedROIS = []
        
        for num in(crop_prediction["rois"].tolist()):
            
            if num not in FreducedROIS:
                FreducedROIS.append(num)
        if FreducedROIS[0][0] == 0 and FreducedROIS[0][1] == 0 and FreducedROIS[0][2] == 0 and FreducedROIS[0][3] == 0: 
           FreducedROIS = []
        
        FreducedCLASSIDS = [76]*len(FreducedROIS)
        FreducedSCORES = [1]*len(FreducedROIS)
        
        
        crop_prediction["class_ids"] = np.array(FreducedCLASSIDS)
        crop_prediction["scores"] = np.array(FreducedSCORES)
        crop_prediction["rois"] = np.array(FreducedROIS)
        
        


if __name__ == '__main__':
    run()

        
        
        
        

