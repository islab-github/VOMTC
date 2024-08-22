# VOMTC: Vision Objects for Millimeter and Terahertz Communications

![](VOMTC_github.png)

Recent advances in sensing and computer vision (CV) technologies have opened the door for the application of
deep learning (DL)-based CV technologies in the realm of 6G wireless communications. For the successful application of this
emerging technology, it is crucial to have a qualified vision dataset tailored for wireless applications (e.g., RGB images
containing wireless devices such as laptops and cell phones). An aim of this work is to propose a large-scale vision dataset
referred to as Vision Objects for Millimeter and Terahertz Communications (VOMTC). The VOMTC dataset consists of
20,232 pairs of RGB and depth images obtained from a camera attached to the base station (BS), with each pair labeled with
three representative object categories (person, cell phone, and  laptop) and bounding boxes of the objects. Through experimental
studies of the VOMTC datasets, we show that the beamforming technique exploiting the VOMTC-trained object detector outper
forms conventional beamforming techniques.

For more details of this work, see the paper "VOMTC: Vision Objects for Millimeter and Terahertz Communications", to appear in IEEE Transactions on Cognitive Communications and Networking http://isl.snu.ac.kr/publication.


## Preparation

Please download the VOMTC training, validation, and test sets from https://drive.google.com/drive/folders/1pmFc7Y7_asoUdh5guweV0FStL_yn1kEq?usp=drive_link and organize the data accordingly (i.e., distance information and RGB/depth images in the 'image' folder and labels in the 'label' folder). 

Also, please download the VOMTC-V2 set from https://drive.google.com/drive/folders/1QyyTzwwiZl1E4RK_Zsiy3TRYryfMWaEK?usp=sharing

Download efficientdet-d8.pth from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch and save it in ./EfficientDet/logs/coco.

## VOMTC dataset selection code

See VOMTCdatasetSelection.py.
Note that this py file is a function file so that it needs to be imported 
by another python file for its execution (in our case, TrainDataforWirelessDeviceDetector.py). 

## Training Data Label Code for the VOMTC-based object detector

Run TrainDataforWielessDeviceDetector.py to obtain input-output pairs for training the VOMTC-based object detector.
After running the code, cropped images and labels will be available in the folder named 'cropped'. 

## To Train the VOMTC-based object detector

python3 trainWirelessDeviceDetector.py --head_only True --load_weights ./logs/coco/efficientdet-d8.pth

## To Test the VOMTC-based object detector on the VOMTC validation set

python3 validateWirelessDeviceDetector.py

## To Test the VOMTC-based object detector on the VOMTC test set

python3 testWirelessDeviceDetector.py

## To Test the VOMTC-based object detector on the VOMTC-V2 dataset

python3 testWirelessDeviceDetector_VOMTCv2.py
