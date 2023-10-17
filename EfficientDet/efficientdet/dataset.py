import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import xml.etree.ElementTree as ElemTree
import json
from collections import OrderedDict

LABEL_DIR = "/home/glen/VisionWireless"
DATA_DIR = "/home/glen/VisionWireless"

### GENERATED DATA FROM VOTRE
def load_generated_data():
    data = OrderedDict()
    
    for root, dirs, files in os.walk(f"{DATA_DIR}/cropped/RGB_cropped"):
        for file in files:
            if not file.endswith(".png"):
                continue
            
            rgb_path = f"{DATA_DIR}/cropped/RGB_cropped/{file}"
            #depth_path = f"{DATA_DIR}/cropped/depth_cropped/{file}"
            CLASSID = f"{DATA_DIR}/cropped/FTFCN_dataLabel/CLASSID/{file}".replace(".png", ".npy")
            INDEX = f"{DATA_DIR}/cropped/FTFCN_dataLabel/INDEX/{file}".replace(".png", ".npy")
            ROIS= f"{DATA_DIR}/cropped/FTFCN_dataLabel/ROIS/{file}".replace(".png", ".npy")
            SCORE =  f"{DATA_DIR}/cropped/FTFCN_dataLabel/SCORE/{file}".replace(".png", ".npy")
            
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            #assert os.path.isfile(depth_path), f"Depth {depth_path} does not exist."
            assert os.path.isfile(CLASSID), f"Class ID {CLASSID} does not exist."
            assert os.path.isfile(rgb_path), f"Index {INDEX} does not exist."
           
            assert os.path.isfile(CLASSID), f"Score {SCORE} does not exist."


            rgb_key = file
            
            #data[rgb_key] = (rgb_path, depth_path, CLASSID, INDEX, ROIS, SCORE)
            data[rgb_key] = (rgb_path, CLASSID, INDEX, ROIS, SCORE)
            
    for root, dirs, files in os.walk(f"{DATA_DIR}/cropped/RGB_cropped2"):
        for file in files:
            if not file.endswith(".png"):
                continue
            
            rgb_path = f"{DATA_DIR}/cropped/RGB_cropped2/{file}"
            #depth_path = f"{DATA_DIR}/cropped/depth_cropped/{file}"
            CLASSID = f"{DATA_DIR}/cropped/FTFCN_dataLabel2/CLASSID/{file}".replace(".png", ".npy")
            INDEX = f"{DATA_DIR}/cropped/FTFCN_dataLabel2/INDEX/{file}".replace(".png", ".npy")
            ROIS= f"{DATA_DIR}/cropped/FTFCN_dataLabel2/ROIS/{file}".replace(".png", ".npy")
            SCORE =  f"{DATA_DIR}/cropped/FTFCN_dataLabel2/SCORE/{file}".replace(".png", ".npy")
            
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            #assert os.path.isfile(depth_path), f"Depth {depth_path} does not exist."
            assert os.path.isfile(CLASSID), f"Class ID {CLASSID} does not exist."
            assert os.path.isfile(rgb_path), f"Index {INDEX} does not exist."
           
            assert os.path.isfile(CLASSID), f"Score {SCORE} does not exist."


            rgb_key = file
            
            #data[rgb_key] = (rgb_path, depth_path, CLASSID, INDEX, ROIS, SCORE)
            data[rgb_key] = (rgb_path, CLASSID, INDEX, ROIS, SCORE)
            
    for root, dirs, files in os.walk(f"{DATA_DIR}/cropped/RGB_cropped3"):
        for file in files:
            if not file.endswith(".png"):
                continue
            
            rgb_path = f"{DATA_DIR}/cropped/RGB_cropped3/{file}"
            #depth_path = f"{DATA_DIR}/cropped/depth_cropped/{file}"
            CLASSID = f"{DATA_DIR}/cropped/FTFCN_dataLabel3/CLASSID/{file}".replace(".png", ".npy")
            INDEX = f"{DATA_DIR}/cropped/FTFCN_dataLabel3/INDEX/{file}".replace(".png", ".npy")
            ROIS= f"{DATA_DIR}/cropped/FTFCN_dataLabel3/ROIS/{file}".replace(".png", ".npy")
            SCORE =  f"{DATA_DIR}/cropped/FTFCN_dataLabel3/SCORE/{file}".replace(".png", ".npy")
            
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            #assert os.path.isfile(depth_path), f"Depth {depth_path} does not exist."
            assert os.path.isfile(CLASSID), f"Class ID {CLASSID} does not exist."
            assert os.path.isfile(rgb_path), f"Index {INDEX} does not exist."
           
            assert os.path.isfile(CLASSID), f"Score {SCORE} does not exist."


            rgb_key = file
            
            #data[rgb_key] = (rgb_path, depth_path, CLASSID, INDEX, ROIS, SCORE)
            data[rgb_key] = (rgb_path, CLASSID, INDEX, ROIS, SCORE)
            
    for root, dirs, files in os.walk(f"{DATA_DIR}/cropped/RGB_cropped4"):
        for file in files:
            if not file.endswith(".png"):
                continue
            
            rgb_path = f"{DATA_DIR}/cropped/RGB_cropped4/{file}"
            #depth_path = f"{DATA_DIR}/cropped/depth_cropped/{file}"
            CLASSID = f"{DATA_DIR}/cropped/FTFCN_dataLabel4/CLASSID/{file}".replace(".png", ".npy")
            INDEX = f"{DATA_DIR}/cropped/FTFCN_dataLabel4/INDEX/{file}".replace(".png", ".npy")
            ROIS= f"{DATA_DIR}/cropped/FTFCN_dataLabel4/ROIS/{file}".replace(".png", ".npy")
            SCORE =  f"{DATA_DIR}/cropped/FTFCN_dataLabel4/SCORE/{file}".replace(".png", ".npy")
            
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            #assert os.path.isfile(depth_path), f"Depth {depth_path} does not exist."
            assert os.path.isfile(CLASSID), f"Class ID {CLASSID} does not exist."
            assert os.path.isfile(rgb_path), f"Index {INDEX} does not exist."
           
            assert os.path.isfile(CLASSID), f"Score {SCORE} does not exist."


            rgb_key = file
            
            #data[rgb_key] = (rgb_path, depth_path, CLASSID, INDEX, ROIS, SCORE)
            data[rgb_key] = (rgb_path, CLASSID, INDEX, ROIS, SCORE)
    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data

def load_generated_data_partial():
    data = OrderedDict()
    
    for root, dirs, files in os.walk(f"{DATA_DIR}/cropped/RGB_cropped2"):
        for file in files:
            if not file.endswith(".png"):
                continue
            
            rgb_path = f"{DATA_DIR}/cropped/RGB_cropped2/{file}"
            #depth_path = f"{DATA_DIR}/cropped/depth_cropped/{file}"
            CLASSID = f"{DATA_DIR}/cropped/FTFCN_dataLabel2/CLASSID/{file}".replace(".png", ".npy")
            INDEX = f"{DATA_DIR}/cropped/FTFCN_dataLabel2/INDEX/{file}".replace(".png", ".npy")
            ROIS= f"{DATA_DIR}/cropped/FTFCN_dataLabel2/ROIS/{file}".replace(".png", ".npy")
            SCORE =  f"{DATA_DIR}/cropped/FTFCN_dataLabel2/SCORE/{file}".replace(".png", ".npy")
            
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            #assert os.path.isfile(depth_path), f"Depth {depth_path} does not exist."
            assert os.path.isfile(CLASSID), f"Class ID {CLASSID} does not exist."
            assert os.path.isfile(rgb_path), f"Index {INDEX} does not exist."
           
            assert os.path.isfile(CLASSID), f"Score {SCORE} does not exist."


            rgb_key = file
            
            #data[rgb_key] = (rgb_path, depth_path, CLASSID, INDEX, ROIS, SCORE)
            data[rgb_key] = (rgb_path, CLASSID, INDEX, ROIS, SCORE)
            
    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data

    
   
    
   
  
            
   
   
   
    
   
            
    
            
    
   


#def parse_xml(xml_path: str, target: str = "phone"):
#    assert target in ("person", "phone")
#    tree = ElemTree.parse(xml_path)
#    objects = tree.findall("object")
#    name = [x.findtext("name") for x in objects]

#    result = []
#    for i, object_iter in enumerate(objects):
#        if ("P" in name[i]) and (target == "phone"):  # phone
#            box = object_iter.find("bndbox")  # noqa
#            result.append([int(it.text) for it in box])  # (x1, y1, x2, y2)
            
            
            
#        elif ("P" not in name[i]) and (target == "person"):  # person
#            box = object_iter.find("bndbox")  # noqa
#            result.append([int(it.text) for it in box])  # (x1, y1, x2, y2)
            
            
#    return result
def parse_distance(json_path: str) -> np.ndarray:
    with open(json_path, "r") as f:
        dist = json.load(f)

    # key = "y_x" , y [0, 480), x [0, 640)
    result = np.zeros((480, 640), dtype=np.float32)
    for k, v in dist.items():
        y, x = k.split("_")
        result[int(y), int(x)] = float(v)
    return result

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
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


def preprocess(*image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas




    


class VisionWirelessDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        #self.set_name = set
        #self.transform = transform
        
        #self.data_name = imageset
        self.coco = COCO(os.path.join('/home/glen/EfficientDet/data/coco/annotations/instances_val2017.json'))
        
        #self.image_ids = self.coco.getImgIds() #COCO dataset의 고유의 것
        self.data = load_generated_data_partial()
        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        
       
        #rgb_path, img = self.load_image(idx)
        rgb_path = self.load_image(idx)
        depth_path, depthimg = self.load_depth_image(idx)
        gt_path, person_annot, phone_annot = self.load_annotations(idx)
        #sample = {'rgb_path': rgb_path , 'img': img, 'depthimg': depthimg, 'annot': annot}
        sample = {'depthimg': depthimg, 'person_annot': person_annot, 'phone_annot': phone_annot}
        #if self.transform:
        #    sample = self.transform(sample)
        sample.update({'rgb_path': rgb_path})
        sample.update({'depth_path': depth_path})
        sample.update({'gt_path': gt_path})
        return sample

    def load_image(self, idx):
        #image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        
        #path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        
        #path = os.path.join(self.root_dir, self.data_name, image_info['file_name'])
        
        
        rgb_path = list(self.data.items())[idx][1][1]
            
        
        
        
        
        
        #img = cv2.imread(rgb_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return rgb_path #, img.astype(np.float32) / 255.
    
    def load_depth_image(self, idx):
        depth_path = list(self.data.items())[idx][1][2]
        
        #ori_img2 = parse_distance(distance_path)
        
        #depth_path = f"depthdata/{'depth' + str(idx) + '.png'}"
               
    
        #plt.imsave(depth_path, ori_img2)
        
       
        return depth_path #, ori_img2 
        #return norm_img[0]
    def load_annotations(self, idx):
        
        
        xml_path = list(self.data.items())[idx][1][0]
        
        tree = ElemTree.parse(xml_path)
        objects = tree.findall("object")
        name = [x.findtext("name") for x in objects]

        person_annotations = np.zeros((1,4))
        phone_annotations = np.zeros((1,4))
        
        
        for j, object_iter in enumerate(objects):
            
            if "person" in name[j]:
                box = object_iter.find("bndbox")  
                annotation = [int(it.text) for it in box]
                #x1, y1, x2, y2 = annotation
                #print((x1, y1, x2, y2))
                annotation = np.array(annotation).reshape(1,4)
                
                person_annotations = np.append(person_annotations, annotation, axis = 0)
                
            elif "P" in name[j]:
                box = object_iter.find("bndbox")  
                annotation = [int(it.text) for it in box]
                #xphone1, yphone1, xphone2, yphone2 = annotation
                #print((xphone1, yphone1, xphone2, yphone2))
                annotation = np.array(annotation).reshape(1,4)
                
                phone_annotations = np.append(phone_annotations, annotation, axis = 0)
                
        person_annotations = person_annotations[1:]
        phone_annotations = phone_annotations[1:]
        
        
        #tree = ElemTree.parse(xml_path)
        #objects = tree.findall("object")
        #name = [x.findtext("name") for x in objects]

        #annotations = np.zeros((0,5))
        #for i, object_iter in enumerate(objects):
        #    if ("P" in name[i]): # and "L" not in name[i] and "person" not in name[i]):  # phone
        #        box = object_iter.find("bndbox")  # noqa
        #        annotation = np.zeros((1,5))
        #        annotation[0, :4] = [int(it.text) for it in box]# (x1, y1, x2, y2)
        #        annotation[0,4] = 76
        #        annotations = np.append(annotations, annotation, axis = 0)
            
            
            
        #    elif ("person" in name[i]): #and "L" not in name[i] and "P" not in name[i]):  # person
        #        box = object_iter.find("bndbox")  # noqa
        #        annotation = np.zeros((1,5))
        #        annotation[0, :4] = [int(it.text) for it in box] # (x1, y1, x2, y2)
        #        annotation[0, 4] = 0
        #        annotations = np.append(annotations, annotation, axis = 0)
        
            #elif ("L" in name[i] and "P" not in name[i] and "person" not in name[i]):  # laptop
            #else:
            #    box = object_iter.find("bndbox")  # noqa
            #    annotation = np.zeros((1,5))
            #    annotation[0, :4] = [int(it.text) for it in box]# (x1, y1, x2, y2)
            #    annotation[0,4] = 72
            #    annotations = np.append(annotations, annotation, axis = 0)
        # get ground truth annotations
        #annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        #annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        #if len(annotations_ids) == 0:
        #    return annotations

        # parse annotations
        #coco_annotations = self.coco.loadAnns(annotations_ids)
        #for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
        #    if a['bbox'][2] < 1 or a['bbox'][3] < 1:
        #        continue

       #     annotation = np.zeros((1, 5))
       #     annotation[0, :4] = a['bbox']
       #     annotation[0, 4] = a['category_id'] - 1
       #     annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        #annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        #annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return xml_path, person_annotations, phone_annotations
    

    
    
    
class CocoDataset(Dataset):
    def __init__(self, root_dir, set, imageset, transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform
        
        self.data_name = imageset
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        
        #path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        
        path = os.path.join(self.root_dir, self.data_name, image_info['file_name'])
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

class VisionWirelessCroppedDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        
        #self.set_name = set
        #self.transform = transform
        
        #self.data_name = imageset
        self.coco = COCO(os.path.join('/home/glen/EfficientDet/data/coco/annotations/instances_val2017.json'))
        
        #self.image_ids = self.coco.getImgIds() #COCO dataset의 고유의 것
        self.data = load_generated_data()
        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        
       
        #rgb_path, img = self.load_image(idx)
        rgb_imgs = self.load_image(idx)
        depth_imgs = self.load_depth_image(idx)
        CLASSID, INDEX, ROIS, SCORE = self.load_annotations(idx)
        sample = {'rgb_path': rgb_imgs, 'depth_path': depth_imgs, 'CLASSID': CLASSID , 'INDEX': INDEX, 'ROIS': ROIS, 'SCORE': SCORE}
        
        #if self.transform:
        #    sample = self.transform(sample)
        #sample.update({'rgb_path': rgb_path})
        #sample.update({'depth_path': depth_path})
        
        return sample

    def load_image(self, idx):
        
        
        
        rgb_path = list(self.data.items())[idx][1][0]
        
        
        
        
        
      
        return rgb_path #, img.astype(np.float32) / 255.
    
    def load_depth_image(self, idx):
        depth_path = list(self.data.items())[idx][1][1]
        
        
       
        return depth_path #img_th
        #return norm_img[0]
    def load_annotations(self, idx):
        
        
        CLASSID = list(self.data.items())[idx][1][1]
        INDEX = list(self.data.items())[idx][1][2]
        ROIS = list(self.data.items())[idx][1][3]
        SCORE  = list(self.data.items())[idx][1][4]
        
       

        return CLASSID, INDEX, ROIS, SCORE
    
def collater(data):
    #imgs = [s['img'] for s in data]
    
    CLASSIDs = []
    for s in data:
        depth_path = s['CLASSID']
        CLASSIDs.append(depth_path)
    INDEXs = []
    for s in data:
        gt_path = s['INDEX']
        INDEXs.append(gt_path)
    ROISs = []
    for s in data:
        file_name = s['ROIS']
        ROISs.append(file_name)
    SCOREs = []
    for s in data:
        file_name = s['SCORE'] 
        SCOREs.append(file_name) 
    rgb_imgs = []
    for s in data:
        file_name = s['rgb_path'] 
        rgb_imgs.append(file_name) 
    
    depth_imgs = []
    for s in data:
        file_name = s['depth_path'] 
        depth_imgs.append(file_name) 
    #rgb_paths = [s['rgb_path'] for s in data]
    
    
    
    #for s in data:
    #    person_annot = np.array(s['person_annot'])
        
    #    if np.size(person_annot) >= 1:
    #        person_annots.append(person_annot)
    #    else:
    #        person_annots.append(np.zeros((1,4)))
    #for i, s in enumerate(data):
    #    phone_annot = np.array(s['phone_annot'])
        
    #    if np.size(phone_annot) >= 1:
        
    #        phone_annots.append(phone_annot)
    #    else: 
    #        phone_annots.append(np.zeros((1,4)))
    
    #rgb_imgs = [s['rgb_path'] for s in data]
    #depth_imgs = [s['depth_path'] for s in data]
    
    
    #rgb_imgs = torch.from_numpy(np.stack(rgb_imgs, axis = 0))
    #depth_imgs = torch.from_numpy(np.stack(depth_imgs, axis = 0))
    
    
    
    ##### PERSON ANNOTS #######
    #max_num_annots = max(annot.shape[0] for annot in person_annots)

    #if max_num_annots > 0:

    #    person_annot_padded = torch.ones((len(person_annots), max_num_annots, 5)) * -1

    #    for idx, annot in enumerate(person_annots):
    #        if annot.shape[0] > 0:
    #            person_annot_padded[idx, :annot.shape[0], :] = torch.Tensor(annot)
    #else:
    #    person_annot_padded = torch.ones((len(person_annots), 1, 5)) * -1
    
    ##### PHONE ANNOTS ########
    #max_num_annots = max(annot.shape[0] for annot in phone_annots)

    #if max_num_annots > 0:

    #    phone_annot_padded = torch.ones((len(phone_annots), max_num_annots, 5)) * -1

    #    for idx, annot in enumerate(phone_annots):
    #        if annot.shape[0] > 0:
    #            phone_annot_padded[idx, :annot.shape[0], :] = torch.Tensor(annot)
    #else:
    #    phone_annot_padded = torch.ones((len(phone_annots), 1, 5)) * -1
    
    ############################
    #imgs = imgs.permute(0, 3, 1, 2)
    
    #depth_imgs = depth_imgs.permute(0, 3, 1, 2)
                
    #return {'img': imgs, 'annot': annot_padded, 'scale': scales, 'depthimg': depth_imgs}
    
    return {'rgb_path': rgb_imgs, 'depth_path': depth_imgs, 'CLASSID': CLASSIDs , 'INDEX': INDEXs, 'ROIS': ROISs, 'SCORE': SCOREs}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        
        #image, depth_image, annots = sample['img'], sample['depthimg'], sample['annot']
        
        depth_image, annots = sample['depthimg'], sample['annot']
        
        
        height, width = depth_image.shape
        if height > width:
            scale = self.img_size / height
        #    resized_height = self.img_size
        #    resized_width = int(width * scale)
        else:
            scale = self.img_size / width
        #    resized_height = int(height * scale)
        #    resized_width = self.img_size

        #image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        #new_image = np.zeros((self.img_size, self.img_size, 3))
        #new_image[0:resized_height, 0:resized_width] = image

        ## depth image too 
        #height_d, width_d = depth_image.shape
        #if height_d > width_d:
        #    scale_d = self.img_size / height_d
        #    resized_height_d = self.img_size
        #    resized_width_d = int(width_d * scale_d)
        #else:
        #    scale_d = self.img_size / width_d
        #    resized_height_d = int(height_d * scale_d)
        #    resized_width_d = self.img_size

        #depth_image = cv2.resize(depth_image, (resized_width_d, resized_height_d), interpolation=cv2.INTER_LINEAR)

        #new_image_d = np.zeros((self.img_size, self.img_size))
        #new_image_d[0:resized_height_d, 0:resized_width_d] = depth_image
        
        #####
        annots[:, :4] *= scale

        #return {'img': torch.from_numpy(new_image).to(torch.float32), 'depthimg': depth_image.to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}
        return {'depthimg': torch.from_numpy(depth_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            #image, depth_image, annots = sample['img'], sample['depthimg'], sample['annot']
            depth_image, annots = sample['depthimg'], sample['annot']
            
            
            #image = image[:, ::-1, :]
            #depth_image = depth_image[:, ::-1, :]
            rows, cols = depth_image.shape
            

            ##############
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            ################
            #sample = {'img': image, 'depthimg': depth_image, 'annot': annots}
            sample = {'depthimg': depth_image, 'annot': annots}
        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        #image, depth_image, annots = sample['img'], sample['depthimg'], sample['annot']
        depth_image, person_annots, phone_annots  = sample['depthimg'], sample['person_annot'], sample['phone_annot']
        #return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'depthimg': depth_image, 'annot': annots}
        return {'depthimg': depth_image, 'person_annot': person_annots, 'phone_annot': phone_annots}
    
