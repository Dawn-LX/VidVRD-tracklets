import os
import pickle
import json
from tqdm import tqdm
import torch
import torch.utils.data

from PIL import Image
import cv2
import sys
import numpy as np


from mega_core.structures.bounding_box import BoxList
from mega_core.utils.comm import is_main_process
from mega_core.config import cfg


class VidORDataset(torch.utils.data.Dataset):
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

    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        self.image_set = image_set      # e.g., VidVRD_train_every10frames, 
        self.transforms = transforms

        self.data_dir = data_dir    # data_dir == "datasets"
        self.img_dir = img_dir      # img_dir: e.g.,  datasets/vidor-dataset/train_frames
        self.anno_path = anno_path  # anno_path: e.g., datasets/vidor-dataset/annotation/training
        self.img_index = img_index  # img_index: e.g., datasets/vidor-dataset/img_index/VidORtrain_freq32.txt

        self.is_train = is_train

        self._img_dir = os.path.join(self.img_dir, "%s.JPEG")   # i.e., 对于视频的训练数据，是把视频提取为JPEG图片再做的
        self._anno_path = os.path.join(self.anno_path, "%s/%s.json") # vidvrd-dataset/train/%s/%s.json

        with open(self.img_index) as f:
            lines = [x.strip().split(" ") for x in f.readlines()]  # .strip() 去除字符串首尾的空格

        self.image_set_index = ["%s/%06d" % (x[0], int(x[1])) for x in lines] # e.g., 0000_2401075277/000010   # 代表每张图片的名字
        self.pattern = [x[0] + "/%06d" for x in lines]                        # e.g., 0000_2401075277/%06d    # 代表每个video中图片名字的pattern
        self.frame_seg_id = [int(x[1]) for x in lines]
        self.frame_seg_len = [int(x[2]) for x in lines]

        # self.video_names = [lines[0][0]] #TODO 这种方法对应后面那个TODO
        # for line in lines:
        #     if line[0] != self.video_names[-1]:
        #         self.video_names.append(line[0])

        if self.is_train:
            keep = self.filter_annotation()

            self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
            self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
            self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
            self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]

        self.classes_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.categories = dict(zip(range(len(self.classes)), self.classes))

        if self.is_train:   
            self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_anno.pkl"))  # e.g., VidVRD_train_every10frames_anno.pkl
        else:
            # 在 self.load_annos 中用了 filter_annotation 之后的东西，但是在test的时候是没有filter_annotation的，
            # 所以在 is_train == False 的时候， 就是说 train 和 test 用同一个数据集会出错，index out of range 
            self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_inference.pkl"))


        if not self.is_train:  # from vid_mega.py
            self.start_index = []
            self.start_id = []
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                self.shuffled_index = {}
            for id, image_index in enumerate(self.image_set_index):
                frame_id = int(image_index.split("/")[-1])
                if frame_id == 0:
                    self.start_index.append(id)
                    if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        shuffled_index = np.arange(self.frame_seg_len[id])
                        if cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        self.shuffled_index[str(id)] = shuffled_index

                    self.start_id.append(id)
                else:
                    self.start_id.append(self.start_index[-1])

    def __getitem__(self, idx): # maintain
        if self.is_train:
            return self._get_train(idx)
        else:
            return self._get_test(idx)
        # return self._get_train(idx)

    def _get_train(self, idx): # modified, #已经改好了
        filename = self.image_set_index[idx]    # index 的范围是整个 txt文件的长度
        img = Image.open(self._img_dir % filename).convert("RGB")   # e.g.,  "datasets/vidor-dataset/train_frames/%s.JPEG"  %  "0000_2401075277/000010"

        frame_id = int(filename.split("/")[-1])
        frame_category = 0
        # if frame_id != 0:
        #     frame_category = 1

        # if a video dataset
        img_refs_l = []
        img_refs_m = []
        img_refs_g = []

        # local frames
        offsets = np.random.choice(cfg.MODEL.VID.MEGA.MAX_OFFSET - cfg.MODEL.VID.MEGA.MIN_OFFSET + 1,
                                    cfg.MODEL.VID.MEGA.REF_NUM_LOCAL, replace=False) + cfg.MODEL.VID.MEGA.MIN_OFFSET
        for i in range(len(offsets)):
            ref_id = min(max(self.frame_seg_id[idx] + offsets[i], 0), self.frame_seg_len[idx] - 1)
            ref_filename = self.pattern[idx] % ref_id
            img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
            img_refs_l.append(img_ref)

        # memory frames
        if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
            ref_id_center = max(self.frame_seg_id[idx] - cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL, 0)
            offsets = np.random.choice(cfg.MODEL.VID.MEGA.MAX_OFFSET - cfg.MODEL.VID.MEGA.MIN_OFFSET + 1,
                                        cfg.MODEL.VID.MEGA.REF_NUM_MEM, replace=False) + cfg.MODEL.VID.MEGA.MIN_OFFSET
            for i in range(len(offsets)):
                ref_id = min(max(ref_id_center + offsets[i], 0), self.frame_seg_len[idx] - 1)
                ref_filename = self.pattern[idx] % ref_id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs_m.append(img_ref)

        # global frames
        if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            ref_ids = np.random.choice(self.frame_seg_len[idx], cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL, replace=False)
            for ref_id in ref_ids:
                ref_filename = self.pattern[idx] % ref_id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs_g.append(img_ref)
        
        target = self.get_groundtruth(idx)  # 构建一个 BoxList 对象并返回， target 中包含了这张图片中所有的bbox以及相应的lable
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:  # is_train == False 的时候， self.transforms == None
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

    def _get_test(self, idx): # 已好
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB") # e.g.,  "vidvrd-dataset/images/%s.JPEG"  %  "ILSVRC2015_train_00005003/000010"

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(filename.split("/")[-1])
        frame_category = 0
        if frame_id != 0:
            frame_category = 1

        img_refs_l = []
        # reading other images of the queue (not necessary to be the last one, but last one here)
        ref_id = min(self.frame_seg_len[idx] - 1, frame_id + cfg.MODEL.VID.MEGA.MAX_OFFSET)
        ref_filename = self.pattern[idx] % ref_id
        img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
        img_refs_l.append(img_ref)

        img_refs_g = []
        if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            size = cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == 0 else 1
            shuffled_index = self.shuffled_index[str(self.start_id[idx])]
            for id in range(size):
                temp = (idx - self.start_id[idx] + cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) % self.frame_seg_len[idx]
                # print(temp,len(shuffled_index))
                filename_g = self.pattern[idx] % shuffled_index[temp]
                img = Image.open(self._img_dir % filename_g).convert("RGB")
                img_refs_g.append(img)

        target = self.get_groundtruth(idx)  # 在test的时候target是没有用到的， 但是在测试集的时候，就是直接没有target
        # for box in target.bbox:
        #     print(box.to(torch.int64).tolist(),"bef",target)

        target = target.clip_to_image(remove_empty=True)
        # for box in target.bbox:
        #     print(box.to(torch.int64).tolist(),"aft",target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l
        images["ref_g"] = img_refs_g
        images["frame_category"] = frame_category
        images["seg_len"] = self.frame_seg_len[idx]
        images["pattern"] = self.pattern[idx]       # # e.g., 0000_2401075277/%06d
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms
        images["filename"] = filename

        return images, target, idx


    def __len__(self):  # maintain 
        return len(self.image_set_index)

    def filter_annotation_old(self):  
        # this function has been deprecated
        cache_file =os.path.join(self.cache_dir, self.image_set + "_keep.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                keep = pickle.load(fid)
            if is_main_process():
                print(" keep information loaded from {}".format(cache_file))
            return keep

        keep = np.zeros((len(self)), dtype=np.bool)
        for idx in range(len(self)):
            if idx % 10000 == 0:
                print("Had filtered {} images".format(idx))

            video_frame = self.image_set_index[idx].split('/')  # e.g., 0000_2401075277/000010 idx 的范围是所有视频的帧的总数
            video_name = video_frame[0]                         # e.g., 0000_2401075277
            frame_id = int(video_frame[1])                      # e.g., 10
            with open(self._anno_path % tuple(video_name.split('_')),'r') as json_file:
                video_ann = json.load(json_file)
                # BUG open video_ann for each frame is too time-consuming
            objs = video_ann["trajectories"][frame_id]
            
            # if len(objs) == 0:
            #     import time
            #     print(video_name,frame_id)
            #     print("-="*30)
            #     time.sleep(3600)

            keep[idx] = False if len(objs) == 0 else True
        print("Had filtered {} images".format(len(self)))

        if is_main_process():
            with open(cache_file, "wb") as fid:
                pickle.dump(keep, fid)
            print("Saving keep information into {}".format(cache_file))

        return keep
    
    def filter_annotation(self):  
        cache_file =os.path.join(self.cache_dir, self.image_set + "_keep.pkl")

        if os.path.exists(cache_file):
            print("Loading keep information from {} ... ".format(cache_file))
            with open(cache_file, "rb") as fid:
                keep = pickle.load(fid)
            print("Done.")
            return keep

        video_name_dup_list = [x.split('/')[0] for x in self.image_set_index]   # self.image_set_index is before filtered
        video_name_list = list(set(video_name_dup_list))
        video_name_list.sort(key=video_name_dup_list.index)  # keep the original order
        video_frame_count = {x:video_name_dup_list.count(x) for x in video_name_list}

        keep = np.zeros((len(self)), dtype=bool)
        outer_idx = 0
        print("filtering annotations... ")
        for video_name in tqdm(video_name_list):
            with open(self._anno_path % tuple(video_name.split('_')),'r') as json_file:
                video_ann = json.load(json_file)
            frame_count = video_frame_count[video_name]
            for frame_id in range(frame_count):
                objs = video_ann["trajectories"][frame_id]
                keep[outer_idx] = False if len(objs) == 0 else True
                outer_idx += 1



        with open(cache_file, "wb") as fid:
            pickle.dump(keep, fid)
        print("keep information has been saved into {}".format(cache_file))

        return keep

    def _preprocess_annotation(self, objs,tid2category_map,width_height):
        boxes = []
        gt_classes = []

        im_info = width_height # a tuple of width and height
        for obj in objs:
            bbox = obj["bbox"]
            bbox = [
                np.maximum(float(bbox["xmin"]), 0),
                np.maximum(float(bbox["ymin"]), 0),
                np.minimum(float(bbox["xmax"]), im_info[0] - 1), #NOTE im_info is w,h
                np.minimum(float(bbox["ymax"]), im_info[1] - 1)
            ]
            boxes.append(bbox)
            
            gt_class = tid2category_map[obj["tid"]]
            gt_classes.append(self.classes_to_ind[gt_class])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # print(boxes_tensor.to(torch.int64),"in dataset")
        res = {
            "boxes": boxes_tensor,      #MARK
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res

    def load_annos_old(self, cache_file):
        # this function has been deprecated
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                annos = pickle.load(fid)
            if is_main_process():
                print("annotation information loaded from {}".format(cache_file))
        else:
            annos = []
            for idx in range(len(self)):
                if idx % 10000 == 0:
                    print("Had processed {} images".format(idx))

                video_frame = self.image_set_index[idx].split('/')  # e.g., 0000_2401075277/000010 idx 的范围是所有视频的帧的总数
                video_name = video_frame[0]                         # e.g., 0000_2401075277
                frame_id = int(video_frame[1])                      # e.g., 10
                with open(self._anno_path % tuple(video_name.split('_')),'r') as json_file: 
                    video_ann = json.load(json_file)
                width_height = (int(video_ann["width"]),int(video_ann["height"]))
                objs = video_ann["trajectories"][frame_id]          # 长度为当前frame中bbox的数量
                traj_categories = video_ann["subject/objects"]      # tid 未必从 0 ~ len(traj_categories)-1 都有
                # tid2category_map = [traj["category"] for traj in traj_categories] #  这样写是不对的, tid 未必从 0 ~ len(traj_categories)-1 都有
                tid2category_map = {traj["tid"]:traj["category"] for traj in traj_categories} # 要这样搞
                # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'}

                #TODO 从 with open 到 tid2category_map 这几行应该是每个video处理一次的，但是现在是每个video中的每个frame都执行了一次，有点重复操作
                # BUG open video_ann for each frame is too time-consuming

                anno = self._preprocess_annotation(objs,tid2category_map,width_height)
                annos.append(anno)
            print("Had processed {} images".format(len(self)))

            if is_main_process():
                with open(cache_file, "wb") as fid:
                    pickle.dump(annos, fid)
                print("Saving annotation information into {}".format(cache_file))

        return annos


    def load_annos(self, cache_file):
        if os.path.exists(cache_file):
            print("loading annotation information from {} ... ".format(cache_file))
            with open(cache_file, "rb") as fid:
                annos = pickle.load(fid)
            print("Done.")
            return annos
        
        video_name_dup_list = [x.split('/')[0] for x in self.image_set_index]   # self.image_set_index is after filtered
        video_name_list = list(set(video_name_dup_list))
        video_name_list.sort(key=video_name_dup_list.index)  # keep the original order
        video_frame_count = {x:video_name_dup_list.count(x) for x in video_name_list}

        annos = []
        outer_idx = 0
        print("construct annos... ")
        for video_name in tqdm(video_name_list):
            with open(self._anno_path % tuple(video_name.split('_')),'r') as json_file: 
                video_ann = json.load(json_file)
            width_height = (int(video_ann["width"]),int(video_ann["height"]))
            frame_count = video_frame_count[video_name]
            traj_categories = video_ann["subject/objects"]      # tid 未必从 0 ~ len(traj_categories)-1 都有
            # tid2category_map = [traj["category"] for traj in traj_categories] #  这样写是不对的, tid 未必从 0 ~ len(traj_categories)-1 都有
            tid2category_map = {traj["tid"]:traj["category"] for traj in traj_categories} # 要这样搞
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'}

            for frame_id in range(frame_count):
                objs = video_ann["trajectories"][frame_id]          # 长度为当前frame中bbox的数量
                anno = self._preprocess_annotation(objs,tid2category_map,width_height)
                annos.append(anno)

                outer_idx += 1
        
        print("Saving annotation information into {} ... ".format(cache_file))
        with open(cache_file, "wb") as fid:
            pickle.dump(annos, fid)
        print("Done.")

        return annos

    def get_img_info(self, idx):
        im_info = self.annos[idx]["im_info"]
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
        filename = self.image_set_index[idx]

        img = cv2.imread(self._img_dir % filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        return img, target, filename

    def get_groundtruth(self, idx):
        anno = self.annos[idx]

        width, height = anno["im_info"]  # NOTE im_info is w,h
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"]) 

        return target

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return VidORDataset.classes[class_id]
