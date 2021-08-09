import os
import pickle
import json
import torch
import torch.utils.data

from PIL import Image
import cv2
import sys
import numpy as np


from mega_core.structures.bounding_box import BoxList
from mega_core.utils.comm import is_main_process
from mega_core.config import cfg


class COCOVidORDataset(torch.utils.data.Dataset):
    classes = ['__background__',
        'bread', 'cake', 'dish', 'fruits', 'vegetables', 'crab', 'backpack', 'camera',
        'cellphone', 'handbag', 'laptop', 'suitcase', 'ball/sports_ball', 'bat', 'frisbee',
        'racket', 'skateboard', 'ski', 'snowboard', 'surfboard', 'toy', 'baby_seat', 'bottle',
        'chair', 'cup', 'electric_fan', 'faucet', 'microwave', 'oven', 'refrigerator',
        'screen/monitor', 'sink', 'sofa', 'stool', 'table', 'toilet', 'guitar', 'piano',
        'baby_walker', 'bench', 'stop_sign', 'traffic_light', 'aircraft', 'bicycle', 'bus/truck',
        'car', 'motorcycle', 'scooter', 'train', 'watercraft', 'bird', 'chicken', 'duck', 'penguin',
        'fish', 'stingray', 'crocodile', 'snake', 'turtle', 'antelope', 'bear', 'camel', 'cat', 'cattle/cow',
        'dog', 'elephant', 'hamster/rat', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'pig', 'rabbit',
        'sheep/goat', 'squirrel', 'tiger', 'adult', 'baby', 'child'
    ]
    def __init__(self, image_set, data_dir, img_dir, anno_path, transforms, is_train=True):
        self.image_set = image_set      # e.g., COCO_train_16classes, 
        self.transforms = transforms

        self.data_dir = data_dir    # data_dir == "datasets"
        self.img_dir = img_dir      # img_dir: e.g.,  datasets/COCOdataset/train2014
        self.anno_path = anno_path  # anno_path: e.g., datasets/COCOinVidVRD/COCO_train_16classes.json

        with open(self.anno_path,'rb') as f:
            img_anno_info = pickle.load(f)
        
        self.imgids = img_anno_info["imgids"]  # COCO dataset 中的imgid，未必连续
        self.imgid2names = img_anno_info["imgid2names"]
        self.imgid2wh = img_anno_info["imgid2wh"]
        self.imgid2annos = img_anno_info["imgid2annos"]
        self.cocoCatId2CatName = img_anno_info["cocoCatId2CatName"]

        ## process synonyms 
        synonyms_vidor2coco = img_anno_info["synonyms_vidor2coco"]
        for k in self.cocoCatId2CatName.keys():
            for vidor_name,coco_name in synonyms_vidor2coco.items(): # e.g., "ball/sports_ball":"sports ball",
                if self.cocoCatId2CatName[k] == coco_name:
                    self.cocoCatId2CatName[k] = vidor_name
                    break
    

        

        self.is_train = is_train

        self._img_dir = os.path.join(self.img_dir, "%s")   # datasets/COCOdataset/train2014/%s

        if self.is_train:
            keep = self.filter_annotation()
            self.imgids = [self.imgids[idx] for idx in range(len(keep)) if keep[idx]]
            

        self.classes_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.categories = dict(zip(range(len(self.classes)), self.classes))

        self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_anno.pkl"))  # e.g., COCO_train_16classes_anno.pkl



    def __getitem__(self, idx): # maintain
        if self.is_train:
            return self._get_train(idx)
        else:
            return self._get_test(idx)

    def _get_train(self, idx): 
        img_id = self.imgids[idx]    # 
        filename = self.imgid2names[img_id]
        img = Image.open(self._img_dir % filename).convert("RGB")   # e.g.,  "datasets/COCOdataset/train2014/%s"  %  "COCO_train2014_000000000113.jpg"

        img_refs_l = []
        img_refs_m = []
        img_refs_g = []
        for i in range(cfg.MODEL.VID.MEGA.REF_NUM_LOCAL):
            img_refs_l.append(img.copy())
        if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
            for i in range(cfg.MODEL.VID.MEGA.REF_NUM_MEM):
                img_refs_m.append(img.copy())
        if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            for i in range(cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL):
                img_refs_g.append(img.copy())
        
        target = self.get_groundtruth(idx)  # 构建一个 BoxList 对象并返回， target中包含了这张图片中所有的bbox以及相应的lable
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_m)):
                img_refs_m[i], _ = self.transforms(img_refs_m[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l # local
        images["ref_m"] = img_refs_m # memory
        images["ref_g"] = img_refs_g # global

        return images, target, filename

    def _get_test(self, idx): 
        imgid = self.imgids[idx]  
        filename = self.imgid2names[imgid]  # e.g., COCO_train2014_000000000113.jpg
        img = Image.open(self._img_dir % filename).convert("RGB") # e.g.,  "vidvrd-dataset/images/%s.JPEG"  %  "ILSVRC2015_train_00005003/000010"
        frame_category = 0  # 0 for satrt
        anno = self.annos[idx]
        boxes = anno["boxes"]
        lables = anno["labels"]
        im_info = anno["im_info"]  # w,h
        assert im_info == img.size, "img_info={},img_size={}".format(im_info,img.size)
        target = BoxList(boxes, im_info, mode="xyxy")
        target.add_field("labels", lables)
        ref_l = img.copy()
        ref_g = img.copy()
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            ref_l,_ =self.transforms(ref_l,None)
            ref_g,_ = self.transforms(ref_g,None)
        
        images = {}
        images["cur"] = img
        images["ref_l"] = ref_l
        images["ref_g"] = ref_g
        images["frame_category"] = frame_category
        images["seg_len"] = 1
        images["pattern"] = filename       # # e.g., ILSVRC2015_train_00005003/%06d
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms

        return images, target, idx


    def __len__(self):  #  已改好
        return len(self.imgids)

    def filter_annotation(self):  # 
        cache_file =os.path.join(self.cache_dir, self.image_set + "_keep.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                keep = pickle.load(fid)
            if is_main_process():
                print(" keep information loaded from {}".format(cache_file))
            return keep

        keep = np.zeros((len(self)), dtype=np.bool)
        for idx in range(len(self)):
            if idx % 15000 == 0:
                print("Had filtered {} images".format(idx))
            
            imgid_in_coco = self.imgids[idx]    # the ids in self.imgids is the image_id in COCO, (i.e., NOT 0~len(self)-1 )
            annos = self.imgid2annos[imgid_in_coco]
            iscrowd_list = [anno["iscrowd"] for anno in annos]
            
            keep[idx] = False if all(iscrowd_list) else True   
            # all(...) 全为1返回True， 有一个0就返回False, iscrowd_list 为空时,all(iscrowd_list)也返回True
        print("Had filtered {} images".format(len(self)))

        if is_main_process():
            with open(cache_file, "wb") as fid:
                pickle.dump(keep, fid)
            print("Saving keep information into {}".format(cache_file))

        return keep

    def _preprocess_annotation(self, coco_annos,width_height):
        # coco_annos: annos of all objects in the current image, in COCO's format
        # the format of coco_annos[i]:
        # e.g.,
            # {
            #     'segmentation': [[35.5, 1.76, 13.43, 16.15, 47.02, 19.03, 113.23, 24.79, 172.72, 47.82, 182.31, 51.66, 205.34, 56.45, 226.45, 56.45, 239.89, 57.41, 202.47, 3.68]], 
            #     'area': 6085.1964499999995, 
            #     'iscrowd': 0, 
            #     'image_id': 262145, 
            #     'bbox': [13.43, 1.76, 226.46, 55.65], 
            #     'category_id': 28,   # category id in coco, not in vidvrd
            #     'id': 285569
            # }

        boxes = []
        gt_classes = []
        im_info = tuple(width_height) # a tuple of width and height #TODO 在anno中增加 width height
        for anno in coco_annos:
            if anno["iscrowd"] == 1:
                continue

            xmin,ymin,w,h = anno["bbox"] # 在COCO中， bbox 是 "bbox": [x,y,width,height], x,y 是左上角点
            xmax = xmin + w
            ymax = ymin + h
            bbox = [ 
                np.maximum(float(xmin), 0),
                np.maximum(float(ymin), 0),
                np.minimum(float(xmax), im_info[0] - 1),
                np.minimum(float(ymax), im_info[1] - 1)  # width, height
            ]
            boxes.append(bbox)
            
            class_name = self.cocoCatId2CatName[anno["category_id"]] # anno["category_id"] is the id in coco, not in vidvrd
            gt_classes.append(self.classes_to_ind[class_name])

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),      #MARK
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res

    def load_annos(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                annos = pickle.load(fid)
            if is_main_process():
                print("annotation information loaded from {}".format(cache_file))
        else:
            annos = []
            for idx in range(len(self)):
                if idx % 15000 == 0:
                    print("Had processed {} images".format(idx))

                imgid_in_coco = self.imgids[idx]    # the ids in self.imgids is the image_id in COCO, (i.e., NOT 0~len(self)-1 )

                anno = self._preprocess_annotation(self.imgid2annos[imgid_in_coco],self.imgid2wh[imgid_in_coco])
                annos.append(anno)
            print("Had processed {} images".format(len(self)))

            if is_main_process():
                with open(cache_file, "wb") as fid:
                    pickle.dump(annos, fid)
                print("Saving annotation information into {}".format(cache_file))

        return annos

    def get_img_info(self, idx):
        im_info = self.annos[idx]["im_info"]  # NOTE im_info is wh
        return {"width": im_info[0],"height": im_info[1]}

    @property
    def cache_dir(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_dir = os.path.join(self.data_dir, 'cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    def get_visualization(self, idx):
        filename = self.imgids[idx]

        img = cv2.imread(self._img_dir % filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        return img, target, filename

    def get_groundtruth(self, idx):
        anno = self.annos[idx]

        width, height = anno["im_info"]   # NOTE im_info is wh
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])

        return target

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return COCOVidVRDDataset.classes[class_id]
