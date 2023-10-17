# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 11:56:35 2022

@author: Glen Kim
"""

import os
from collections import OrderedDict



LABEL_VAL_DIR = "/home/glen/VisionWireless/VOTRE/VAL/label"
DATA_VAL_DIR = "/home/glen/VisionWireless/VOTRE/VAL/image"

LABEL_TEST_DIR = "/home/glen/VisionWireless/VOTRE/TEST/label"
DATA_TEST_DIR = "/home/glen/VisionWireless/VOTRE/TEST/image"
#LABEL_DIR = "C:/Users/Glen Kim/VisionWireless/data/label_VOBEM"
#DATA_DIR = "C:/Users/Glen Kim/VisionWireless/data/result_VOBEM"


def load_two_data():
    data = OrderedDict()
    
    file = "cal1_0.png"
    depth_file = "cal1_0_depth_color_image.png"
    directory = 'day13_part1_2'
    gt_path = LABEL_TEST_DIR + "/day13_part1_2/" + file.replace(".png", ".xml")
    rgb_path = DATA_TEST_DIR + "/day13_part1_2/rgb/" + file
    depth_img_path = DATA_TEST_DIR + "/day13_part1_2/depth/" + depth_file
    distance_path = (DATA_TEST_DIR + "/day13_part1_2/distance/" + file).replace(".png", ".json")
    distance_path = distance_path.replace("cal", "cal_distance_")
    
    rgb_key = file.replace(".xml", ".png")
    data[rgb_key] = (gt_path, rgb_path, depth_img_path, distance_path, directory, file)
        
            
    file = "cal4_0.png"
    depth_file = "cal4_0_depth_color_image.png"
    directory = 'day13_part1_2'
    gt_path = LABEL_TEST_DIR + "/day13_part1_2/" + file.replace(".png", ".xml")
    rgb_path = DATA_TEST_DIR + "/day13_part1_2/rgb/" + file
    depth_img_path = DATA_TEST_DIR + "/day13_part1_2/depth/" + depth_file
    distance_path = (DATA_TEST_DIR + "/day13_part1_2/distance/" + file).replace(".png", ".json")
    distance_path = distance_path.replace("cal", "cal_distance_")
    

    rgb_key = file.replace(".xml", ".png")
    data[rgb_key] = (gt_path, rgb_path, depth_img_path, distance_path, directory, file)

    
        
    

    
    
            
    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data


def small_load_data():
    data = OrderedDict()
    
    for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/Day22_part1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day22_part1'
            gt_path = f"{LABEL_VAL_DIR}/Day22_part1/{file}"
            depth_img_path = f"{DATA_VAL_DIR}/Day22_part1/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_VAL_DIR}/Day22_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_VAL_DIR}/Day22_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, depth_img_path, distance_path, directory, original_file)

    
    
    
            
    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data

#
# 수정해야함! 
#


            
def load_test_data():
    data = OrderedDict()
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Testaug"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Testaug'
            gt_path = f"{LABEL_TEST_DIR}/Testaug/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Testaug/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Testaug/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day34"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day34'
            gt_path = f"{LABEL_TEST_DIR}/Day34/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day34/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day34/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day34_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day28_part2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day28_part2'
            gt_path = f"{LABEL_TEST_DIR}/Day28_part2/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day28_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day28_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day28_part3"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day28_part3'
            gt_path = f"{LABEL_TEST_DIR}/Day28_part3/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day28_part3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day28_part3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day28_part4"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day28_part4'
            gt_path = f"{LABEL_TEST_DIR}/Day28_part4/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day28_part4/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day28_part4/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day29"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day29'
            gt_path = f"{LABEL_TEST_DIR}/Day29/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day29/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day29/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day29/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path,  distance_path, directory, original_file)
            
    #for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day30"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day30'
    #        gt_path = f"{LABEL_TEST_DIR}/Day30/{file}"
           
    #        rgb_path = f"{DATA_TEST_DIR}/Day30/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_TEST_DIR}/Day30/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day30_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day32_2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day32_2'
            gt_path = f"{LABEL_TEST_DIR}/Day32_2/{file}"
            
            rgb_path = f"{DATA_TEST_DIR}/Day32_2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day32_2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day32_2_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/DayAdd1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'DayAdd1'
            gt_path = f"{LABEL_TEST_DIR}/DayAdd1/{file}"
           
            distance_path = f"{DATA_TEST_DIR}/DayAdd1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/DayAdd2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'DayAdd2'
            gt_path = f"{LABEL_TEST_DIR}/DayAdd2/{file}"
            
            rgb_path = f"{DATA_TEST_DIR}/DayAdd2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/DayAdd2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/DayAdd3"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'DayAdd3'
            gt_path = f"{LABEL_TEST_DIR}/DayAdd3/{file}"
            
            rgb_path = f"{DATA_TEST_DIR}/DayAdd3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/DayAdd3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    #
    # ORIGINAL TEST VOTRE
    #
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/day13_part1_2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day13_part1_2'
            gt_path = f"{LABEL_TEST_DIR}/day13_part1_2/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/day13_part1_2/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/day13_part1_2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/day13_part1_2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/day13_part1_3"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day13_part1_3'
            gt_path = f"{LABEL_TEST_DIR}/day13_part1_3/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/day13_part1_3/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/day13_part1_3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/day13_part1_3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/day13_part2_2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day14_part1'
            gt_path = f"{LABEL_TEST_DIR}/day13_part2_2/{file}"
            
            rgb_path = f"{DATA_TEST_DIR}/day13_part2_2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/day13_part2_2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/day13_part2_3"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day14_part1'
            gt_path = f"{LABEL_TEST_DIR}/day13_part2_3/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/day13_part2_3/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/day13_part2_3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/day13_part2_3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)     
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day16_part1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day14_part1'
            gt_path = f"{LABEL_TEST_DIR}/Day16_part1/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day16_part1/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day16_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day16_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day16_part2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day14_part1'
            gt_path = f"{LABEL_TEST_DIR}/Day16_part2/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day16_part2/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day16_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day16_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day18_part1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day14_part1'
            gt_path = f"{LABEL_TEST_DIR}/Day18_part1/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day18_part1/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day18_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day18_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path,  distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day19_part1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day14_part1'
            gt_path = f"{LABEL_TEST_DIR}/Day19_part1/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day19_part1/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day19_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day19_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    data = OrderedDict(sorted(data.items()))  
    print(f"Data loaded, total: {len(data)} sets")
    return data


            
def load_test_partial_data():
    data = OrderedDict()
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Testaug"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Testaug'
            gt_path = f"{LABEL_TEST_DIR}/Testaug/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Testaug/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Testaug/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day34"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day34'
            gt_path = f"{LABEL_TEST_DIR}/Day34/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day34/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day34/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day34_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day29"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day29'
            gt_path = f"{LABEL_TEST_DIR}/Day29/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day29/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_TEST_DIR}/Day29/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day29/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path,  distance_path, directory, original_file)
            
    
            
    
    
            
    #for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day32"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day32'
    #        gt_path = f"{LABEL_TEST_DIR}/Day32/{file}"
            
    #        rgb_path = f"{DATA_TEST_DIR}/Day32/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_TEST_DIR}/Day32/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day32_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    #for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day33_2"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day33_2'
    #        gt_path = f"{LABEL_TEST_DIR}/Day33_2/{file}"
            
    #        rgb_path = f"{DATA_TEST_DIR}/Day33_2/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_TEST_DIR}/Day33_2/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day33_2_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    #for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day33"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day33'
    #        gt_path = f"{LABEL_TEST_DIR}/Day33/{file}"
            
    #        rgb_path = f"{DATA_TEST_DIR}/Day33/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_TEST_DIR}/Day33/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day33_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    #for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day35"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day35'
    #        gt_path = f"{LABEL_TEST_DIR}/Day35/{file}"
            
    #        rgb_path = f"{DATA_TEST_DIR}/Day35/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_TEST_DIR}/Day35/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day35_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day30"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day30'
            gt_path = f"{LABEL_TEST_DIR}/Day30/{file}"
            
            rgb_path = f"{DATA_TEST_DIR}/Day30/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day30/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day30_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/Day32_2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day32_2'
            gt_path = f"{LABEL_TEST_DIR}/Day32_2/{file}"
            
            rgb_path = f"{DATA_TEST_DIR}/Day32_2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/Day32_2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day32_2_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/DayAdd1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'DayAdd1'
            gt_path = f"{LABEL_TEST_DIR}/DayAdd1/{file}"
           
            distance_path = f"{DATA_TEST_DIR}/DayAdd1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/DayAdd2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'DayAdd2'
            gt_path = f"{LABEL_TEST_DIR}/DayAdd2/{file}"
            
            rgb_path = f"{DATA_TEST_DIR}/DayAdd2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/DayAdd2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_TEST_DIR}/DayAdd3"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'DayAdd3'
            gt_path = f"{LABEL_TEST_DIR}/DayAdd3/{file}"
            
            rgb_path = f"{DATA_TEST_DIR}/DayAdd3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_TEST_DIR}/DayAdd3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    data = OrderedDict(sorted(data.items()))  
    print(f"Data loaded, total: {len(data)} sets")
    return data  
    
            
def load_val_partial_data():
    data = OrderedDict()
    
    for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/Valaug"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Valaug'
            gt_path = f"{LABEL_VAL_DIR}/Valaug/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_VAL_DIR}/Valaug/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_VAL_DIR}/Valaug/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    data = OrderedDict(sorted(data.items()))  
    print(f"Data loaded, total: {len(data)} sets")
    return data  
    

def load_val_data():
    data = OrderedDict()  
    
    for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/Valaug"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Valaug'
            gt_path = f"{LABEL_VAL_DIR}/Valaug/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_VAL_DIR}/Valaug/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_VAL_DIR}/Valaug/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/Day36"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day36'
            gt_path = f"{LABEL_VAL_DIR}/Day36/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_VAL_DIR}/Day36/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_VAL_DIR}/Day36/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day36_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    #for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/Valaug"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Valaug'
    #        gt_path = f"{LABEL_VAL_DIR}/Valaug/{file}"
            #depth_img_path = f"{DATA_TEST_DIR}/Day34/depth/{file}".replace(".xml", ".png")
    #        rgb_path = f"{DATA_VAL_DIR}/Valaug/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_VAL_DIR}/Valaug/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance_")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    #
    # ORIGINAL VOTRE
    #
    for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/day10_part4"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day10_part4'
            gt_path = f"{LABEL_VAL_DIR}/day10_part4/{file}"
            #depth_img_path = f"{DATA_VAL_DIR}/day10_part4/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_VAL_DIR}/day10_part4/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_VAL_DIR}/day10_part4/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/day11_part4"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day11_part4'
            gt_path = f"{LABEL_VAL_DIR}/day11_part4/{file}"
            #depth_img_path = f"{DATA_VAL_DIR}/day11_part4/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_VAL_DIR}/day11_part4/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_VAL_DIR}/day11_part4/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
           
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path,  distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/Day18_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day18_part2'
            gt_path = f"{LABEL_VAL_DIR}/Day18_part2/{file}"
            #depth_img_path = f"{DATA_VAL_DIR}/Day18_part2/depth/{file}".replace(".xml", ".png")
            rgb_path = f"{DATA_VAL_DIR}/Day18_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_VAL_DIR}/Day18_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
           
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_VAL_DIR}/Day22_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day22_part1'
            gt_path = f"{LABEL_VAL_DIR}/Day22_part1/{file}"
            rgb_path = f"{DATA_VAL_DIR}/Day22_part1/rgb/{file}".replace(".xml", ".png")
            #depth_img_path = f"{DATA_VAL_DIR}/Day22_part1/depth/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_VAL_DIR}/Day22_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
   
            
    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data
