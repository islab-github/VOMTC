# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:41:51 2022

VOMTC dataset Generation code
function parametrized by three parameters: activeClasses, maxnumPeople, maxDist


@author: Glen Kim
"""

import numpy as np
import os
#from dataset import load_data
from VOMTCdataset import load_train_data
from collections import OrderedDict
import json

#
# ADJUSTABLE PARAMETERS:
# 
#activeClasses = [0, 1,2]
#maxnumPeople = 3
#maxDist = 6

def VOMTCdatasetSelection(activeClasses, 
                            maxnumPeople, 
                            maxDist):
    
    def Person_find(i):
        if "person" in txt[i+1]:
            return True
        else:
            return False
    
    def PL_find(i):
        if "P" in txt[i+1]:
            return True
        else:
            return False
        
    def Laptop_find(i):
        if "L" in txt[i+1]:
            return True
        else:
            return False

    detected_without_mobile = []
    detected_without_laptop = []
    detected_without_person = []
    detected_with_mobile_only = []
    detected_with_laptop_only = []
    detected_with_person_only = []
    detected_with_everything = []
    selected_data = OrderedDict()
    selected_data_0 = OrderedDict()
    selected_data_1 = OrderedDict()
    selected_data_2 = OrderedDict()
    selected_data_01 = OrderedDict()
    selected_data_12 = OrderedDict()
    selected_data_02 = OrderedDict()
    selected_data_012 = OrderedDict()
    ## LOAD DATA
    data = load_train_data()
    total = len(data)
    print('ORIGINAL DATA SIZE')
    print(total)
    def parse_distance(json_path: str) -> np.ndarray:
        with open(json_path, "r") as f:
            dist = json.load(f)
        result = np.zeros((480, 640), dtype=np.float32)
        for k, v in dist.items():
            y, x = k.split("_")
            result[int(y), int(x)] = float(v)
        return result
    for count, (data_key, (gt_path, rgb_path, distance_path, directory, name)) in enumerate(data.items()):
        print(distance_path)
        depth = parse_distance(distance_path)
        txt = []
        with open('C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}'.format(directory, name), 'r', encoding='UTF-8') as f:
            for line in f:
                txt.append(line)

        object_start = []
        for i in range(len(txt)):
        # <object>라는 키워드를 찾으면 그 밑 줄을 탐색하면 된다.
        # object_start는 <object>라는 키워드가 있는 줄 집합
        # object_start의 모든 원소+1 줄에서 P나 L이 있는지 찾으면 된다.
            if "<object>" in txt[i]:
                object_start.append(i)

        # <object>라는 키워드가 없어서 사람도, 핸드폰, 랩톱도 없는 경우
        if len(object_start) == 0:
        # 콘솔에 출력한다.
            print("{}/{}".format(directory, name))
            detected_without_mobile.append("{}/{}".format(directory, name))
            detected_without_laptop.append("{}/{}".format(directory, name))
            detected_without_person.append("{}/{}".format(directory, name))
            detected_with_mobile_only.append("{}/{}".format(directory, name))
            detected_with_laptop_only.append("{}/{}".format(directory, name))
            detected_with_person_only.append("{}/{}".format(directory, name))
            detected_with_everything.append("{}/{}".format(directory, name))
            continue

        # PL_in_txt는 핸드폰이 있다는 걸 알려주는 counter
        # Laptop_in_txt는 랩톱이 있다는 걸 알려주는 counter
        # 핸드폰이 하나라도 있으면 True
    
    
        PL_in_txt = False
        Laptop_in_txt = False
        Person_in_txt = False
        person_count = 0
        for i in object_start:
            # 핸드폰이 하나라도 있는 경우
            if PL_find(i):
                #print("There is P or L")
                PL_in_txt = True
            
            
            
            
            # 랩톱이 하나라도 있는 경우
            if Laptop_find(i):
                Laptop_in_txt = True
            
            
            
            # 사람이 하나라도 있는 경우
        
            if Person_find(i) :
                Person_in_txt = True
                person_count = person_count + 1
            
            
    
    
        
        depth_onecol = np.reshape(depth, np.shape(depth)[0]*np.shape(depth)[1])
        ########################################################################
        # ONLY CELL PHONE
        if PL_in_txt == True and Person_in_txt == False and Laptop_in_txt == False  and len(activeClasses) == 1 and activeClasses[0] == 1:
        
            detected_with_mobile_only.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
        
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data_1[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        if Laptop_in_txt == False and PL_in_txt == True and Person_in_txt == False and len(activeClasses) == 1 and activeClasses[0] ==1 and person_count <= maxnumPeople and  max(depth_onecol) <= maxDist:
        
            detected_with_mobile_only.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
            
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        ########################################################################
        # ONLY LAPTOP
        if Laptop_in_txt == True and PL_in_txt == False and Person_in_txt == False and len(activeClasses) == 1 and activeClasses[0] == 2:
        
            detected_with_laptop_only.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
            
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data_2[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        if Laptop_in_txt == True and PL_in_txt == False and Person_in_txt == False and len(activeClasses) == 1 and activeClasses[0] == 2 and person_count <= maxnumPeople and  max(depth_onecol) <= maxDist:
        
            detected_with_laptop_only.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
            
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        #########################################################################
        
        # ONLY PERSON
        if Person_in_txt == True and Laptop_in_txt == False and PL_in_txt == False and len(activeClasses) == 1 and activeClasses[0] == 0:
        
            detected_with_person_only.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data_0[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
            
        if Person_in_txt == True and Laptop_in_txt == False and PL_in_txt == False and len(activeClasses) == 1 and activeClasses[0] == 0 and person_count <= maxnumPeople and  max(depth_onecol) <= maxDist:
            detected_with_person_only.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
       ######################################################################## 
        # 핸드폰 없이 나머지만 있는 경우 [0, 2]
        if Person_in_txt == True and Laptop_in_txt == True and  PL_in_txt == False and len(activeClasses) == 2 and activeClasses[0] == 0 and activeClasses[1] == 2:
        # 콘솔에 출력한다.
        
        
            detected_without_mobile.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data_02[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        if Person_in_txt == True and Laptop_in_txt == True and  PL_in_txt == False and len(activeClasses) == 2 and activeClasses[0] == 0 and activeClasses[1] == 2 and person_count <= maxnumPeople and  max(depth_onecol) <= maxDist:
            detected_without_mobile.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        #####################################################################
        #랩톱 없이 나머지만 있는 경우  [0,1]
        if PL_in_txt == True and Person_in_txt == True and Laptop_in_txt == False and len(activeClasses) == 2 and activeClasses[0] == 0 and activeClasses[1] == 1:
        # 콘솔에 출력한다.
        
        
            detected_without_laptop.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data_01[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        if PL_in_txt == True and Person_in_txt == True and Laptop_in_txt == False and len(activeClasses) == 2 and activeClasses[0] == 0 and activeClasses[1] == 1 and person_count <= maxnumPeople and  max(depth_onecol) <= maxDist:
        # 콘솔에 출력한다.
        
        
            detected_without_laptop.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        #######################################################################
        
        #사람 없이 나머지만 있는 경우  [1,2]
        if Laptop_in_txt == True and PL_in_txt == True and  Person_in_txt == False and len(activeClasses) == 2 and activeClasses[0] == 1 and activeClasses[1] == 2:
        # 콘솔에 출력한다.
        
        
            detected_without_person.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
    
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data_12[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        if Laptop_in_txt == True and PL_in_txt == True and  Person_in_txt == False and len(activeClasses) == 2 and activeClasses[0] == 1 and activeClasses[1] == 2 and person_count <= maxnumPeople and  max(depth_onecol) <= maxDist:
        
        
            detected_without_person.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        
        #######################################################################  
        
        
        
        
            
        
        #[1,2,3]
        if Laptop_in_txt == True and PL_in_txt == True and  Person_in_txt == True and len(activeClasses) == 3 :
        
        
            detected_with_everything.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data_012[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        
        if Laptop_in_txt == True and PL_in_txt == True and  Person_in_txt == True and len(activeClasses) == 3 and person_count <= maxnumPeople and  max(depth_onecol) <= maxDist:
        
        
            detected_with_everything.append("C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label/{}/{}".format(directory, name))
        
            gt_path = gt_path
            rgb_path = rgb_path
            distance_path = distance_path
            
        
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = gt_path.replace(".xml", ".png")
            selected_data[rgb_key] = (gt_path, rgb_path, distance_path, directory, name)
        
        
        
    print(len(selected_data))
    return selected_data
    
# 콘솔에 출력된 것이 아무 것도 없거나 사람만 있는 파일
# detected 에 있는 것도 아무 것도 없거나 사람만 있는 파일

