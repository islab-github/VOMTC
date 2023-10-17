# VOMTC: Vision Objects for Millimeter and Terahertz Communications

![](VOMTC_github.png)

Recent advances in deep learning (DL) and computer vision (CV) have opened the door for the application of DL-based CV technologies in the realm of 6G wireless communications.
Due to the substantial data requirements of DL-based CV, it is crucial to construct a qualified vision dataset tailored for wireless applications (e.g.,  RGB images containing wireless devices such as laptops and cell phones).
In this paper, we propose a large-scale vision dataset referred to as Vision Objects for Millimeter and Terahertz and Environment (VOMTC) designed for CV-based wireless applications.
The VOMTC dataset consists of 20,232 pairs of RGB and depth images obtained from a camera attached to the base station (BS), with each pair labeled with three representative object categories (person, cell phone, and laptop) and bounding boxes of the objects.
To facilitate researchers in selecting the inputs and outputs that align with their wireless application, we design VOMTC with the following three key parameters: 1) active classes, 2) maximum number of people, and 3) maximum distance to the farthest object.
Through experimentation using the VOMTC validation and test datasets, we demonstrate that the object detector model fine-tuned using VOTRE outperforms the baseline object detector in identifying cell phones.

For more details of this work, see the paper "VOMTC: Vision Objects for Millimeter and Terahertz Communications", submitted to IEEE Transactions on Cognitive Communications.


## Preparation

Please download the VOMTC training, validation, and test sets from https://www.dropbox.com/sh/qmq0hulzrprnc0z/AAAwcycHwQ9KA8NTPlcMLwFIa?dl=0 and organize the data accordingly (i.e., distance information and RGB/depth images in the 'image' folder and labels in the 'label' folder). 

Download efficientdet-d8.pth from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch and save it in ./EfficientDet/logs/coco.

## VOTRE dataset selection code

See VOMTCdatasetSelection.py.
Note that this py file is a function file so that it needs to be imported 
by another python file for its execution (in our case, TrainDataforWirelessDeviceDetector.py). 

## Train Data Label Generation for training the VOTRE-based object detector

Run TrainDataforWielessDeviceDetector.py to obtain input-output pairs for training the VOMTC-based object detector.
After running the code, cropped images and labels will be available in the folder named 'cropped'. 

## To Train the VOTRE-based object detector

python3 trainWirelessDeviceDetector.py --head_only True --load_weights ./logs/coco/efficientdet-d8.pth

## To Test the VOTRE-based object detector on the VOTRE validation set

python3 validateWirelessDeviceDetector_Failedsampleincluded.py

## To Test the VOTRE-based object detector on the VOTRE test set

python3 testWirelessDeviceDetector_Failedsampleincluded.py
