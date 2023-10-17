import torch
import torch.nn as nn
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
#from utils.utils import postprocess, invert_affine, display
from utils2 import postprocess, invert_affine, display
import torch.nn.functional as F
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:2")
device3 = torch.device("cuda:3")


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

class FocalLossReg(nn.Module):
    def __init__(self):
        super(FocalLossReg, self).__init__()

    def forward(self, regressions, anchors, annotations, **kwargs):
        
        batch_size = regressions.shape[0]
        #classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            #classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            #bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            #classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            ##my code
            
            
            
            ######################
            
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    
                    #alpha_factor = torch.ones_like(classification) * alpha
                    #alpha_factor = alpha_factor.cuda()
                    #alpha_factor = 1. - alpha_factor
                    #focal_weight = classification
                    #focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    #bce = -(torch.log(1.0 - classification))
                    
                    #cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    #regression_losses.append(torch.tensor(0).to(dtype))
                    #classification_losses.append(cls_loss.sum())
                else:
                    
                    #alpha_factor = torch.ones_like(classification) * alpha
                    #alpha_factor = 1. - alpha_factor
                    #focal_weight = classification
                    #focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    #bce = -(torch.log(1.0 - classification))
                    
                    #cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype))
                    #classification_losses.append(cls_loss.sum())

                continue
            #print('---------------------------Anchor and BBOX ---------------------------')
            #print(anchor[:,:].size())    
            #print(bbox_annotation[:, :4].size())
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
            
            
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            
            positive_indices = torch.ge(IoU_max, 0.5)
            

            assigned_annotations = bbox_annotation[IoU_argmax, :]
            #print('---------------------------IoU_max---------------------------')
            #print(IoU_max.size())
            
            # compute the loss for classification
            
            
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    #regression_losses.append(torch.tensor(0).to(dtype))
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            #regressBoxes = BBoxTransform()
            #clipBoxes = ClipBoxes()
            #obj_list = kwargs.get('obj_list', None)
            #out = postprocess(imgs.detach(),
            #                  torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
            #                  regressBoxes, clipBoxes,
            #                  0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            #display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50 
    
class FocalLossCla(nn.Module):
    def __init__(self):
        super(FocalLossCla, self).__init__()

    def forward(self, classifications,  annotations, lengths, **kwargs):
        
        batch_size = classifications.shape[0]
        classification_losses = []
        #regression_losses = []

        
        #anchor_widths = anchor[:, 3] - anchor[:, 1]
        #anchor_heights = anchor[:, 2] - anchor[:, 0]
        #anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        #anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights
        
        
        for j in range(batch_size):

            classification = classifications[j, :, :]
            targets = annotations[j, : ,:]
            #positive_indices = pos_indices_set[j]
            #bbox_annotation = annotations[j]
            #bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            classification = classification[:lengths[j]]
            targets = targets[:lengths[j]]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4).to(device1)
            #classification = classification.to(device1)
            
            ##my code
            
            #classification0 = torch.reshape(classification[:, 0], ( len(classification[:, 0]), 1))
           
            #classification76 = torch.reshape(classification[:, 76], ( len(classification[:, 76]),1))
            
            #classification = torch.cat((classification0, classification76), 1)
            #classification = classification76
            ###############################
            
            #classification0 = f_classification[:,0]
            #classification76 = f_classification[:,76]
            #classification = torch.zeros(f_classification.size()[0], f_classification.size()[1])
            #classification[:, 0] = classification0
            #classification[:, 76] = classification76
            #classification = classification.cuda()
            #del classification0
            #del classification76
            #del f_classification
            #torch.cuda.empty_cache()
            #classification0 = np.reshape(f_classification[:,0], (len(f_classification[:,0]), 1))
            #classification76 = np.reshape(f_classification[:,76], (len(f_classification[:,76]), 1))
            
            
            #classification = torch.zeros_like(classification)
            #classification = torch.empty(classification.shape[0] , 2)
            
            ## numpy to GPU Tensor
            #f_classification = f_classification.detach().cpu().numpy()
            #classes = np.zeros([f_classification.shape[0], f_classification.shape[1]])
            
            #classification0 = f_classification[:,0]
            #classification76 = f_classification[:,76]
            #classes[:, 0] = classification0
            #classes[:, 76] = classification76
                      
            #classification = torch.Tensor(classes).cuda()
            ######################
            
            #if bbox_annotation.shape[0] == 0:
            #if torch.cuda.is_available():
                    
            #    alpha_factor = torch.ones_like(classification) * alpha
            #    alpha_factor = alpha_factor.to(device1)
            #    alpha_factor = 1. - alpha_factor
            #    focal_weight = classification.to(device1)
            #    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
            #    bce = -(torch.log(1.0 - classification)).to(device1)
                    
            #    cls_loss = focal_weight * bce
                    
                   
            #    classification_losses.append(cls_loss.sum())
            #else:
                    
            #    alpha_factor = torch.ones_like(classification) * alpha
            #    alpha_factor = 1. - alpha_factor
            #    focal_weight = classification
            #    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
            #    bce = -(torch.log(1.0 - classification))
                    
            #    cls_loss = focal_weight * bce
                    
                    
            #    classification_losses.append(cls_loss.sum())

            #continue
            
            
            # compute the loss for classification
            
            #targets = torch.ones_like(classification) * 0 # -1
           
            #assign = bbox_annotation
            
            #if torch.cuda.is_available():
            #    targets = targets.to(device1)
            
            
            
           
            
            
            
            
            #for i in range(len(assign)):
                
                    
            #    if assign[i] == 76:
            #       targets[i,1] = 1
                   
            #    else:
            #       targets[i,0] = 1
                   
              
            
            
            
            
            
            cls_loss = F.binary_cross_entropy(classification, targets)
            
            classification_losses.append(cls_loss)

            
            #if positive_indices.sum() > 0:
            #    assigned_annotations = assigned_annotations[positive_indices, :]

            #    anchor_widths_pi = anchor_widths[positive_indices]
            #    anchor_heights_pi = anchor_heights[positive_indices]
            #    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
            #    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

            #    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
            #    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
            #    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
            #    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
            #    gt_widths = torch.clamp(gt_widths, min=1)
            #    gt_heights = torch.clamp(gt_heights, min=1)

            #    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
            #    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
            #    targets_dw = torch.log(gt_widths / anchor_widths_pi)
            #    targets_dh = torch.log(gt_heights / anchor_heights_pi)

            #    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
            #    targets = targets.t()

            #    regression_diff = torch.abs(targets - regression[positive_indices, :])

            #    regression_loss = torch.where(
            #        torch.le(regression_diff, 1.0 / 9.0),
            #        0.5 * 9.0 * torch.pow(regression_diff, 2),
            #        regression_diff - 0.5 / 9.0
            #    )
            #    regression_losses.append(regression_loss.mean())
            #else:
            #    if torch.cuda.is_available():
            #        regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    #regression_losses.append(torch.tensor(0).to(dtype))
            #    else:
            #        regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        #imgs = kwargs.get('imgs', None)
        #if imgs is not None:
            #regressBoxes = BBoxTransform()
            #clipBoxes = ClipBoxes()
            #obj_list = kwargs.get('obj_list', None)
            #out = postprocess(imgs.detach(),
            #                  torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
            #                  regressBoxes, clipBoxes,
            #                  0.5, 0.3)
            #imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            #imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            #imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            #display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(classification_losses).sum(dim=0, keepdim=True)
    
    


  
          
           