# -*- coding: utf-8 -*-
"""
VOTRE dataset

@author: S. Kim
"""

import os
from collections import OrderedDict

LABEL_DIR = "C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/label"
DATA_DIR = "C:/Users/Glen Kim/VisionWireless/VOMTC/TRAIN/image"

#LABEL_VAL_DIR = "C:/Users/Glen Kim/VisionWireless/VOMTC/VAL/label"
#DATA_VAL_DIR = "C:/Users/Glen Kim/VisionWireless/VOMTC/VAL/image"

#LABEL_TEST_DIR = "C:/Users/Glen Kim/VisionWireless/VOMTC/TEST/label"
#DATA_TEST_DIR = "C:/Users/Glen Kim/VisionWireless/VOMTC/TEST/image"



def load_train_data_2():
    
    data = OrderedDict()  
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day28_part1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day28_part1'
            gt_path = f"{LABEL_DIR}/Day28_part1/{file}"
            rgb_path = f"{DATA_DIR}/Day28_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day28_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day31"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day31'
    #        gt_path = f"{LABEL_DIR}/Day31/{file}"
    #        rgb_path = f"{DATA_DIR}/Day31/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day31/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day31_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day32"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day32'
    #        gt_path = f"{LABEL_DIR}/Day32/{file}"
    #        rgb_path = f"{DATA_DIR}/Day32/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day32/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day32_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day32_2"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day32_2'
    #        gt_path = f"{LABEL_DIR}/Day32_2/{file}"
    #        rgb_path = f"{DATA_DIR}/Day32_2/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day32_2/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day32_2_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day33"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day33'
    #        gt_path = f"{LABEL_DIR}/Day33/{file}"
    #        rgb_path = f"{DATA_DIR}/Day33/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day33/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day33_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day33_2"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day33_2'
    #        gt_path = f"{LABEL_DIR}/Day33_2/{file}"
    #        rgb_path = f"{DATA_DIR}/Day33_2/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day33_2/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #       distance_path = distance_path.replace("Day33_2_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day35"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day35'
    #        gt_path = f"{LABEL_DIR}/Day35/{file}"
    #        rgb_path = f"{DATA_DIR}/Day35/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day35/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day35_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day35_partial"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day35'
    #        gt_path = f"{LABEL_DIR}/Day35_partial/{file}"
    #        rgb_path = f"{DATA_DIR}/Day35/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day35/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day35_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Trainaug"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Trainaug'
    #        gt_path = f"{LABEL_DIR}/Trainaug/{file}"
    #        rgb_path = f"{DATA_DIR}/Trainaug/rgb/{file}".replace(".xml", ".png")
            
            
    #        distance_path = f"{DATA_DIR}/Trainaug/distance/{file}".replace(".xml", ".json")
    #        file = file.replace("part1_cal", "part1_cal_distance_")
    #        file = file.replace("part2_cal", "part2_cal_distance_")
    #        file = file.replace("indoor2_820_cal", "indoor2_820_cal_distance_")
    #        file = file.replace("indoor_820_cal", "indoor_820_cal_distance_")
            
    #        file = file.replace("day1_301_cal", "day1_301_cal_distance_")
    #        file = file.replace("day1_2_301_cal", "day1_2_301_cal_distance_")
    #        file = file.replace("day10_part3_cal", "day10_part3_cal_distance_")
    #        file = file.replace("day9_night302_1_cal", "day9_night302_1_cal_distance_")
    #        file = file.replace("day9_night302_2_cal", "day9_night302_2_cal_distance_")
    #        distance_path = f"{DATA_DIR}/Trainaug/distance/{file}".replace(".xml", ".json")
            
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
       
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Trainaug2"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Trainaug2'
    #        gt_path = f"{LABEL_DIR}/Trainaug2/{file}"
    #        rgb_path = f"{DATA_DIR}/Trainaug2/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Trainaug2/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
            
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day37"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day37'
    #        gt_path = f"{LABEL_DIR}/Day37/{file}"
    #        rgb_path = f"{DATA_DIR}/Day37/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day37/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day37_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Day38"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Day38'
    #        gt_path = f"{LABEL_DIR}/Day38/{file}"
    #        rgb_path = f"{DATA_DIR}/Day38/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Day38/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day38_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    #for root, dirs, files in os.walk(f"{LABEL_DIR}/Set8"):
    #    for file in files:
    #        if not file.endswith(".xml"):
    #            continue
    #        original_file = file
    #        directory = 'Set8'
    #        gt_path = f"{LABEL_DIR}/Set8/{file}"
    #        rgb_path = f"{DATA_DIR}/Set8/rgb/{file}".replace(".xml", ".png")
    #        distance_path = f"{DATA_DIR}/Set8/distance/{file}".replace(".xml", ".json")
    #        distance_path = distance_path.replace("cal", "cal_distance")
    #        distance_path = distance_path.replace("Day36_", "")
    #        assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
    #        assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
    #        assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

    #        rgb_key = file.replace(".xml", ".png")
    #        data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data
    


def load_train_data():
    
    data = OrderedDict()  
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day1_2_301_part1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'day1_2_301_part1'
            gt_path = f"{LABEL_DIR}/day1_2_301_part1/{file}"
            rgb_path = f"{DATA_DIR}/day1_2_301_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day1_2_301_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day1_2_301_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day1_2_301_part2'
            gt_path = f"{LABEL_DIR}/day1_2_301_part2/{file}"
            rgb_path = f"{DATA_DIR}/day1_2_301_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day1_2_301_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day1_301_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day1_301_part1'
            gt_path = f"{LABEL_DIR}/day1_301_part1/{file}"
            rgb_path = f"{DATA_DIR}/day1_301_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day1_301_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day1_301_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day1_301_part2'
            gt_path = f"{LABEL_DIR}/day1_301_part2/{file}"
            rgb_path = f"{DATA_DIR}/day1_301_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day1_301_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
           
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day2_2_302_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day2_2_302_part1'
            gt_path = f"{LABEL_DIR}/day2_2_302_part1/{file}"
            rgb_path = f"{DATA_DIR}/day2_2_302_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day2_2_302_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day2_302_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day2_302_part1'
            gt_path = f"{LABEL_DIR}/day2_302_part1/{file}"
            rgb_path = f"{DATA_DIR}/day2_302_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day2_302_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day3_2_301_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day3_2_301_part1'
            gt_path = f"{LABEL_DIR}/day3_2_301_part1/{file}"
            rgb_path = f"{DATA_DIR}/day3_2_301_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day3_2_301_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day3_301_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day3_301_part1'
            gt_path = f"{LABEL_DIR}/day3_301_part1/{file}"
            rgb_path = f"{DATA_DIR}/day3_301_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day3_301_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day4_2_engbuilding_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day4_2_engbuilding_part1'
            gt_path = f"{LABEL_DIR}/day4_2_engbuilding_part1/{file}"
            rgb_path = f"{DATA_DIR}/day4_2_engbuilding_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day4_2_engbuilding_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day4_engbuilding_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day4_engbuilding_part1'
            gt_path = f"{LABEL_DIR}/day4_engbuilding_part1/{file}"
            rgb_path = f"{DATA_DIR}/day4_engbuilding_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day4_engbuilding_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day5_2_301_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day5_2_301_part1'
            gt_path = f"{LABEL_DIR}/day5_2_301_part1/{file}"
            rgb_path = f"{DATA_DIR}/day5_2_301_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day5_2_301_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day5_301_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day5_301_part1'
            gt_path = f"{LABEL_DIR}/day5_301_part1/{file}"
            rgb_path = f"{DATA_DIR}/day5_301_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day5_301_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day6_2_chemical_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day6_2_chemical_part1'
            gt_path = f"{LABEL_DIR}/day6_2_chemical_part1/{file}"
            rgb_path = f"{DATA_DIR}/day6_2_chemical_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day6_2_chemical_part1/distance/{file}".replace(".xml", ".json")
            file = file.replace("part1_cal", "part1_cal_distance_")
            distance_path = f"{DATA_DIR}/day6_2_chemical_part1/distance/{file}".replace(".xml", ".json")
            #distance_path = distance_path.replace("cal_distance_","cal")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day6_chemical_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day6_chemical_part1'
            gt_path = f"{LABEL_DIR}/day6_chemical_part1/{file}"
            rgb_path = f"{DATA_DIR}/day6_chemical_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day6_chemical_part1/distance/{file}".replace(".xml", ".json")
            file = file.replace("part1_cal", "part1_cal_distance_")
            distance_path = f"{DATA_DIR}/day6_chemical_part1/distance/{file}".replace(".xml", ".json")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day7_othersidesofcampus1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day7_othersidesofcampus1'
            gt_path = f"{LABEL_DIR}/day7_othersidesofcampus1/{file}"
            rgb_path = f"{DATA_DIR}/day7_othersidesofcampus1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day7_othersidesofcampus1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day7_othersidesofcampus2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day7_othersidesofcampus2'
            gt_path = f"{LABEL_DIR}/day7_othersidesofcampus2/{file}"
            rgb_path = f"{DATA_DIR}/day7_othersidesofcampus2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day7_othersidesofcampus2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day7_othersidesofcampus3"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day7_othersidesofcampus3'
            gt_path = f"{LABEL_DIR}/day7_othersidesofcampus3/{file}"
            rgb_path = f"{DATA_DIR}/day7_othersidesofcampus3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day7_othersidesofcampus3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
           
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day7_othersidesofcampus4"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day7_othersidesofcampus4'
            gt_path = f"{LABEL_DIR}/day7_othersidesofcampus4/{file}"
            rgb_path = f"{DATA_DIR}/day7_othersidesofcampus4/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day7_othersidesofcampus4/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day8_night301_1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day8_night301_1'
            gt_path = f"{LABEL_DIR}/day8_night301_1/{file}"
            rgb_path = f"{DATA_DIR}/day8_night301_1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day8_night301_1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day8_night301_2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day8_night301_2'
            gt_path = f"{LABEL_DIR}/day8_night301_2/{file}"
            rgb_path = f"{DATA_DIR}/day8_night301_2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day8_night301_2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day8_night301_3"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day8_night301_3'
            gt_path = f"{LABEL_DIR}/day8_night301_3/{file}"
            rgb_path = f"{DATA_DIR}/day8_night301_3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day8_night301_3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day8_night301_4"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day8_night301_4'
            gt_path = f"{LABEL_DIR}/day8_night301_4/{file}"
            rgb_path = f"{DATA_DIR}/day8_night301_4/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day8_night301_4/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day9_night302_1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day9_night302_1'
            gt_path = f"{LABEL_DIR}/day9_night302_1/{file}"
            rgb_path = f"{DATA_DIR}/day9_night302_1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day9_night302_1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
        
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day9_night302_2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day9_night302_2'
            gt_path = f"{LABEL_DIR}/day9_night302_2/{file}"
            rgb_path = f"{DATA_DIR}/day9_night302_2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day9_night302_2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
     
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day10_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day10_part1'
            gt_path = f"{LABEL_DIR}/day10_part1/{file}"
            rgb_path = f"{DATA_DIR}/day10_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day10_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
         
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day10_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day10_part2'
            gt_path = f"{LABEL_DIR}/day10_part2/{file}"
            rgb_path = f"{DATA_DIR}/day10_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day10_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day10_part3"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day10_part3'
            gt_path = f"{LABEL_DIR}/day10_part3/{file}"
            rgb_path = f"{DATA_DIR}/day10_part3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day10_part3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day11_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day11_part1'
            gt_path = f"{LABEL_DIR}/day11_part1/{file}"
            rgb_path = f"{DATA_DIR}/day11_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day11_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day11_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day11_part2'
            gt_path = f"{LABEL_DIR}/day11_part2/{file}"
            rgb_path = f"{DATA_DIR}/day11_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day11_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day11_part3"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day11_part3'
            gt_path = f"{LABEL_DIR}/day11_part3/{file}"
            rgb_path = f"{DATA_DIR}/day11_part3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day11_part3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day12_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day12_part1'
            gt_path = f"{LABEL_DIR}/day12_part1/{file}"
            rgb_path = f"{DATA_DIR}/day12_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day12_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day12_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day12_part2'
            gt_path = f"{LABEL_DIR}/day12_part2/{file}"
            rgb_path = f"{DATA_DIR}/day12_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day12_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day13_part1_1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day13_part1_1'
            gt_path = f"{LABEL_DIR}/day13_part1_1/{file}"
            rgb_path = f"{DATA_DIR}/day13_part1_1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day13_part1_1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day13_part2_1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day13_part2_1'
            gt_path = f"{LABEL_DIR}/day13_part2_1/{file}"
            rgb_path = f"{DATA_DIR}/day13_part2_1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day13_part2_1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day14_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day14_part1'
            gt_path = f"{LABEL_DIR}/day14_part1/{file}"
            rgb_path = f"{DATA_DIR}/day14_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day14_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day14_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day14_part2'
            gt_path = f"{LABEL_DIR}/day14_part2/{file}"
            rgb_path = f"{DATA_DIR}/day14_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day14_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day15_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day15_part1'
            gt_path = f"{LABEL_DIR}/day15_part1/{file}"
            rgb_path = f"{DATA_DIR}/day15_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day15_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
           
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day15_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day15_part2'
            gt_path = f"{LABEL_DIR}/day15_part2/{file}"
            rgb_path = f"{DATA_DIR}/day15_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day15_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day15_part3"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day15_part3'
            gt_path = f"{LABEL_DIR}/day15_part3/{file}"
            rgb_path = f"{DATA_DIR}/day15_part3/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day15_part3/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/day15_part4"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'day15_part4'
            gt_path = f"{LABEL_DIR}/day15_part4/{file}"
            rgb_path = f"{DATA_DIR}/day15_part4/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/day15_part4/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day17_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day17_part1'
            gt_path = f"{LABEL_DIR}/Day17_part1/{file}"
            rgb_path = f"{DATA_DIR}/Day17_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day17_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day17_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day17_part2'
            gt_path = f"{LABEL_DIR}/Day17_part2/{file}"
            rgb_path = f"{DATA_DIR}/Day17_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day17_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day19_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day19_part2'
            gt_path = f"{LABEL_DIR}/Day19_part2/{file}"
            rgb_path = f"{DATA_DIR}/Day19_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day19_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day20_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day20_part1'
            gt_path = f"{LABEL_DIR}/Day20_part1/{file}"
            rgb_path = f"{DATA_DIR}/Day20_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day20_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day20_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day20_part2'
            gt_path = f"{LABEL_DIR}/Day20_part2/{file}"
            rgb_path = f"{DATA_DIR}/Day20_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day20_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day21_part1"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day21_part1'
            gt_path = f"{LABEL_DIR}/Day21_part1/{file}"
            rgb_path = f"{DATA_DIR}/Day21_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day21_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day21_part2"):
        for file in files:
            if not file.endswith(".xml"):
               continue
            original_file = file
            directory = 'Day21_part2'
            gt_path = f"{LABEL_DIR}/Day21_part2/{file}"
            rgb_path = f"{DATA_DIR}/Day21_part2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day21_part2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day28_part1"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day28_part1'
            gt_path = f"{LABEL_DIR}/Day28_part1/{file}"
            rgb_path = f"{DATA_DIR}/Day28_part1/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day28_part1/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day31"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day31'
            gt_path = f"{LABEL_DIR}/Day31/{file}"
            rgb_path = f"{DATA_DIR}/Day31/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day31/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day31_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day32"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day32'
            gt_path = f"{LABEL_DIR}/Day32/{file}"
            rgb_path = f"{DATA_DIR}/Day32/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day32/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day32_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day32_2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day32_2'
            gt_path = f"{LABEL_DIR}/Day32_2/{file}"
            rgb_path = f"{DATA_DIR}/Day32_2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day32_2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day32_2_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day33"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day33'
            gt_path = f"{LABEL_DIR}/Day33/{file}"
            rgb_path = f"{DATA_DIR}/Day33/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day33/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day33_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day33_2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day33_2'
            gt_path = f"{LABEL_DIR}/Day33_2/{file}"
            rgb_path = f"{DATA_DIR}/Day33_2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day33_2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day33_2_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day35"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day35'
            gt_path = f"{LABEL_DIR}/Day35/{file}"
            rgb_path = f"{DATA_DIR}/Day35/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day35/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day35_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Trainaug"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Trainaug'
            gt_path = f"{LABEL_DIR}/Trainaug/{file}"
            rgb_path = f"{DATA_DIR}/Trainaug/rgb/{file}".replace(".xml", ".png")
            
            
            distance_path = f"{DATA_DIR}/Trainaug/distance/{file}".replace(".xml", ".json")
            file = file.replace("part1_cal", "part1_cal_distance_")
            file = file.replace("part2_cal", "part2_cal_distance_")
            file = file.replace("indoor2_820_cal", "indoor2_820_cal_distance_")
            file = file.replace("indoor_820_cal", "indoor_820_cal_distance_")
            
            file = file.replace("day1_301_cal", "day1_301_cal_distance_")
            file = file.replace("day1_2_301_cal", "day1_2_301_cal_distance_")
            file = file.replace("day10_part3_cal", "day10_part3_cal_distance_")
            file = file.replace("day9_night302_1_cal", "day9_night302_1_cal_distance_")
            file = file.replace("day9_night302_2_cal", "day9_night302_2_cal_distance_")
            distance_path = f"{DATA_DIR}/Trainaug/distance/{file}".replace(".xml", ".json")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
       
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Trainaug2"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Trainaug2'
            gt_path = f"{LABEL_DIR}/Trainaug2/{file}"
            rgb_path = f"{DATA_DIR}/Trainaug2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Trainaug2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day37"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day37'
            gt_path = f"{LABEL_DIR}/Day37/{file}"
            rgb_path = f"{DATA_DIR}/Day37/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day37/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day37_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day38"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day38'
            gt_path = f"{LABEL_DIR}/Day38/{file}"
            rgb_path = f"{DATA_DIR}/Day38/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day38/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day38_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
    
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Set8"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Set8'
            gt_path = f"{LABEL_DIR}/Set8/{file}"
            rgb_path = f"{DATA_DIR}/Set8/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Set8/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day36_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    for root, dirs, files in os.walk(f"{LABEL_DIR}/Day30"):
        for file in files:
            if not file.endswith(".xml"):
                continue
            original_file = file
            directory = 'Day30'
            gt_path = f"{LABEL_DIR}/Day30/{file}"
            rgb_path = f"{DATA_DIR}/Day30/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/Day30/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance")
            distance_path = distance_path.replace("Day30_", "")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."
            
            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, original_file)
            
    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data
    

