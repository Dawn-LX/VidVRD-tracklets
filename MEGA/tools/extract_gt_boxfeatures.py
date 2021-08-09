import os
import pickle
import json
import argparse
from tqdm import tqdm
from collections import defaultdict,deque
from PIL import Image

import numpy as np
np.set_printoptions(precision=4,suppress=True,linewidth=1000)
import torch

from mega_core.structures.image_list import to_image_list
from mega_core.structures.boxlist_ops import cat_boxlist
from mega_core.structures.bounding_box import BoxList
from mega_core.modeling.backbone.backbone import build_backbone
from mega_core.modeling.roi_heads.roi_heads import build_roi_heads

from tools.categories_v2 import vidor_CatName2Id,vidor_CatId2name

class FeatureExtractor(torch.nn.Module):
    def __init__(self,cfg,image_set_index,annos,filtered_frame_idx):
        super(FeatureExtractor,self).__init__()
        self.device = cfg.MODEL.DEVICE

        self.backbone = build_backbone(cfg)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.memory_enable = cfg.MODEL.VID.MEGA.MEMORY.ENABLE
        self.global_enable = cfg.MODEL.VID.MEGA.GLOBAL.ENABLE

        self.base_num = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N
        self.advanced_num = int(self.base_num * cfg.MODEL.VID.MEGA.RATIO)

        self.all_frame_interval = cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL  # 25
        self.key_frame_location = cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION  # 12

        self.image_set_index = image_set_index  # list[str] each str is similar as '0000_2401075277/000010'   # 代表每张图片的名字
        self.annos = annos
        self.filtered_frame_idx = filtered_frame_idx
    
    
    def get_groundtruth(self, filename):
        idx = self.image_set_index.index(filename)
        anno = self.annos[idx]

        width, height = anno["im_info"]  # NOTE im_info is w,h
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"]) 
        target.add_field("tids", anno["tids"])

        return target

    def forward(self, images):
        """
        Arguments:
            images = {}
            images["cur"] = (img,target)    # img 是tensor ， transforms 的最后有to_tensor 的操作
            images["ref_l"] = img_refs_l    # list[tuple], each one is (img_ref,target_ref)
            images["ref_g"] = img_refs_g    # list[tuple], each one is (img_g,target_g)
            images["frame_category"] = frame_category
            images["seg_len"] = self.frame_seg_len[idx]
            images["pattern"] = self.pattern[idx]       # # e.g., 0000_2401075277/%06d
            images["img_dir"] = self._img_dir
            images["transforms"] = self.transforms
            images["filename"] = filename
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        assert not self.training
        
        images["cur"] = (to_image_list(images["cur"][0]), images["cur"][1])
        images["ref_l"] = [(to_image_list(img),tgt) for img,tgt in images["ref_l"]]
        images["ref_g"] = [(to_image_list(img),tgt) for img,tgt in images["ref_g"]]
        
        infos = images.copy()
        infos.pop("cur")
        return self._forward_test(images["cur"], infos)

    def _forward_test(self, img_tgt, infos):
        """
        forward for the test phase.
        :param img_tgt: (img,target)
        :param infos:
        :param targets:
        :return:
        """
        
        def update_feature(img=None, feats=None, proposals=None, proposals_feat=None):
            assert (img is not None) or (feats is not None and proposals is not None and proposals_feat is not None)
            assert proposals != None, "please input gt target as proposals to extract gt features"
            if img is not None:
                feats = self.backbone(img)[0]
                # note here it is `imgs`! for we only need its shape, it would not cause error, but is not explicit.
                # proposals = self.rpn(imgs, (feats,), version="ref")  # rpn 返回的是一个 list， list of BoxList, len == batch_size
                
                proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)

            self.feats.append(feats)
            self.proposals.append(proposals[0])     
            self.proposals_dis.append(proposals[0][:self.advanced_num]) # BoxList 的对象是可以这样取值的， 有重载 __getitem__ 函数， 这里就是取前几个
            self.proposals_feat.append(proposals_feat)
            self.proposals_feat_dis.append(proposals_feat[:self.advanced_num])

        
        if infos["frame_category"] == 0:  # a new video
            # self.seg_len = infos["seg_len"]  # 这个是经过 dataset 里的 filter_annotation 之后的
            self.end_id = 0
            self.video_name = infos["pattern"].split('/')[0]
            self.frame_ids= self.filtered_frame_idx[self.video_name]

            self.feats = deque(maxlen=self.all_frame_interval)
            self.proposals = deque(maxlen=self.all_frame_interval)
            self.proposals_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_dis = deque(maxlen=self.all_frame_interval)

            self.roi_heads.box.feature_extractor.init_memory()
            if self.global_enable:
                self.roi_heads.box.feature_extractor.init_global()

            img_cur,tgt_cur = img_tgt
            feats_cur = self.backbone(img_cur.tensors)[0]  # ResNet-C4 只有一个 feature-level
            # proposals_cur = self.rpn(imgs, (feats_cur, ), version="ref") # rpn 返回的是一个 list， list of BoxList, len == batch_size
            proposals_cur = [tgt_cur]
            proposals_feat_cur = self.roi_heads.box.feature_extractor(feats_cur, proposals_cur, pre_calculate=True)
            while len(self.feats) < self.key_frame_location + 1:
                update_feature(None, feats_cur, proposals_cur, proposals_feat_cur)

            while len(self.feats) < self.all_frame_interval:  
                self.end_id = min(self.end_id + 1, len(self.frame_ids) - 1)  
                end_filename = infos["pattern"] % self.frame_ids[self.end_id]
                end_image = Image.open(infos["img_dir"] % end_filename).convert("RGB")
                end_target = self.get_groundtruth(end_filename)
                #Problem: 如果 end_filename 不在 self.image_set_index 中怎么办, i.e, 它被 filter_annotation 过滤掉了， 怎么办？
                # 把 self.filtered_frame_idx 传进来
                end_image,end_target = infos["transforms"](end_image,end_target)  # transforms 有 to tensor 的操作
                if isinstance(end_image, tuple):
                    end_image = end_image[0]
                end_image = end_image.view(1, *end_image.shape).to(img_cur.tensors.device)
                end_target = end_target.to(end_image.device)

                update_feature(end_image,proposals=[end_target])

        elif infos["frame_category"] == 1:
            self.end_id = min(self.end_id + 1, len(self.frame_ids) - 1)
            end_image,end_target = infos["ref_l"][0]  # ref_l 里面直接取出来的已经经过 transforms 了, 这里的[0]因为它是一个len==1的list
            # end_image 是一个 ImageList 的对象
            end_image = end_image.tensors
            update_feature(end_image,proposals=[end_target])

        # 1. update global
        if infos["ref_g"]:
            for global_img,global_tgt in infos["ref_g"]:
                feats = self.backbone(global_img.tensors)[0]
                proposals = [global_tgt]
                proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)

                self.roi_heads.box.feature_extractor.update_global(proposals_feat)

        feats = self.feats[self.key_frame_location]
        
        img_cur,tgt_cur = img_tgt
        num_box = tgt_cur.bbox.shape[0]
        # proposals, proposal_losses = self.rpn(imgs, (feats, ), None)
        proposals = [tgt_cur]

        proposals_ref = cat_boxlist(list(self.proposals))
        proposals_ref_dis = cat_boxlist(list(self.proposals_dis))
        proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)
        proposals_feat_ref_dis = torch.cat(list(self.proposals_feat_dis), dim=0)

        proposals_list = [proposals, proposals_ref, proposals_ref_dis, proposals_feat_ref, proposals_feat_ref_dis]

        if self.roi_heads:
            # x, result, detector_losses = self.roi_heads(feats, proposals_list, None) # 原来是这样的
            # 但是现在我们不是过整个 roi_heads， 而是只需要 过其中的 feature_extractor
            box_features = self.roi_heads.box.feature_extractor(feats, proposals_list)
            assert num_box == box_features.shape[0], " box_features.shape = {}".format(box_features.shape)

        else:
            assert False, "please set roi_heads"

        return box_features


class VidORDatasetGt(object):
    """"this class is used to extract box_features of gt tracklets"""

    def __init__(self, cfg, image_set, data_dir, img_dir, anno_path, img_index, transforms,exclude_video_list=[]):
        self.is_train = False
        self.cfg = cfg
        self.image_set = image_set      # e.g., VidVRD_train_every10frames, 
        self.transforms = transforms  
        # test 的时候也是有 transform的，包括ColorJitter和像素的 Normalize， 
        # 不然的话不适配网络的权重。所以直接之前写的把图片输入到backbone是不对的

        self.data_dir = data_dir    # data_dir == "datasets"
        self.img_dir = img_dir      # img_dir: e.g.,  datasets/vidor-dataset/train_frames
        self.anno_path = anno_path  # anno_path: e.g., datasets/vidor-dataset/annotation/training
        self.img_index = img_index  # img_index: e.g., datasets/vidor-dataset/img_index/VidORtrain_freq32.txt

        self.exclude_video_list = exclude_video_list


        self._img_dir = os.path.join(self.img_dir, "%s.JPEG")   # i.e., 对于视频的训练数据，是把视频提取为JPEG图片再做的
        self._anno_path = os.path.join(self.anno_path, "%s/%s.json") # vidvrd-dataset/train/%s/%s.json

        with open(self.img_index) as f:
            lines = [x.strip().split(" ") for x in f.readlines()]  # .strip() 去除字符串首尾的空格

        self.image_set_index = ["%s/%06d" % (x[0], int(x[1])) for x in lines] # e.g., 0000_2401075277/000010   # 代表每张图片的名字
        self.pattern = [x[0] + "/%06d" for x in lines]                        # e.g., 0000_2401075277/%06d    # 代表每个video中图片名字的pattern
        self.frame_seg_id = [int(x[1]) for x in lines]
        self.frame_seg_len = [int(x[2]) for x in lines]


        # NOTE 我们现在是队gt跑test，所以还是要用到anno的

        keep = self.filter_annotation()
        self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
        self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
        self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
        self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]
        self._prepare_video_frame_infos()

        self.Name2Id = vidor_CatName2Id
    

        self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_anno.pkl"))  # e.g., VidVRD_train_every10frames_anno.pkl
        # self.annos is a list，其中每一个item 和 image_set_index中每个img是对应的。 顺序是保持对应的
        assert len(self.annos) == len(self.image_set_index)
        assert len(set(self.image_set_index)) == len(self.image_set_index)

        # # 在 self.load_annos 中用了 filter_annotation 之后的东西，但是在test的时候是没有filter_annotation的，
            # # 所以在 is_train == False 的时候， 就是说 train 和 test 用同一个数据集会出错，index out of range 
            # self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_inference.pkl"))


        if not self.is_train:  # from vid_mega.py
            self.start_index = []
            self.start_id = []
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                self.shuffled_index = {}
            for id, image_index in enumerate(self.image_set_index):
                video_name,frame_id= image_index.split("/")
                frame_ids_list = self.filtered_frame_idx[video_name]
                start_frame_id = frame_ids_list[0]

                frame_id = int(frame_id)
                # if frame_id == 0:  
                #     如果 frame_id == 0 这一帧被过滤掉了怎么办？ 那这个视频不是被当成上一个视频了吗？（即，这个视频的每一帧记录的 start_id 都是上一个视频的start_id）
                if frame_id == start_frame_id:
                    self.start_index.append(id)  # 这个 id 是全局的，所以start_id 就是记录每个video是从哪个id开始的
                    if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        # shuffled_index = np.arange(self.frame_seg_len[id]) # shuffled_index 里存的是 frame_id
                        # frame_seg_len[id] 的长度，是 filter 之前的长度，我们现在要搞成 filter之后的，把filter掉的不算进去
                        # 在原来的代码中，这样是没问题的，就是说，在原来的代码中，shuffled_index 中的 index 对应的图片，可以没有 annotation，因为 proposal是用 rpn提的
                        # 但是在我们这里，把gt的标注作为 proposal， 所以应该 shuffled_index 中的每一个index对应的图片都要有 annotation
                        # 所以我们的 shuffled_index 应该设置为过滤之后的 frame_id
                        shuffled_index = np.array(frame_ids_list)
                        if cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        self.shuffled_index[str(id)] = shuffled_index  # 每一个起始帧，都有一个 shuffled_index, 
                        # self.shuffled_index[video_name] = shuffled_index # 等价于每个视频都有一个 shuffled_index, 这样写可读性更强 （但是呢，原来的代码等不动就尽量不动，万一有点什么问题）

                    self.start_id.append(id)
                else:
                    self.start_id.append(self.start_index[-1])

    def __getitem__(self, idx): # maintain
        if self.exclude_video_list == []:
            return self._get_test(idx)
        

        video_name,frame_id = self.image_set_index[idx].split("/")
        if video_name in self.exclude_video_list:
            return None,idx
        else:
            return self._get_test(idx)

        

    def _prepare_video_frame_infos(self):
        cache_file =os.path.join(self.cache_dir, self.image_set + "_frame_infos.pkl")

        if os.path.exists(cache_file):
            print("Loading keep information from {} ... ".format(cache_file))
            with open(cache_file, "rb") as fid:
                frame_infos = pickle.load(fid)
            (
                self.video_name_list,
                self.video_frame_count,
                self.filtered_frame_idx
            ) = frame_infos
            print("Done.")
            return 
        
        print("preparing filtered video frame ids ...")
        video_name_dup_list = [x.split('/')[0] for x in self.image_set_index]   # self.image_set_index is after filtered
        self.video_name_list = list(set(video_name_dup_list))
        self.video_name_list.sort(key=video_name_dup_list.index)  # keep the original order
        self.video_frame_count = {x:video_name_dup_list.count(x) for x in self.video_name_list}

        self.filtered_frame_idx = {}
        for video_name in tqdm(self.video_name_list):
            temp = [x.split('/') for x in self.image_set_index] # e.g., x == "0000_2401075277/000010"
            frame_ids = sorted([int(x[1]) for x in temp if x[0] == video_name])
            
            self.filtered_frame_idx[video_name] = frame_ids

        frame_infos = (
            self.video_name_list,
            self.video_frame_count,
            self.filtered_frame_idx
        )
        with open(cache_file, "wb") as fid:
            pickle.dump(frame_infos, fid)
        print("frame_infos has been saved into {}".format(cache_file))


    def _get_test(self, idx): # 已好
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB") # e.g.,  "vidvrd-dataset/images/%s.JPEG"  %  "ILSVRC2015_train_00005003/000010"

        # give the current frame a category. 0 for start, 1 for normal
        video_name,frame_id = filename.split("/")
        frame_id = int(frame_id)
        frame_ids_list = self.filtered_frame_idx[video_name]
        start_frame_id = frame_ids_list[0]

        frame_category = 0
        if frame_id != start_frame_id:  # start frame
            frame_category = 1

        img_refs_l = []
        # reading other images of the queue (not necessary to be the last one, but last one here)
        # ref_id = min(self.frame_seg_len[idx] - 1, frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET)
        # frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET 这一帧不一定在 frame_ids_list 里面
        ref_relative_id = min(len(frame_ids_list)-1, frame_ids_list.index(frame_id) + self.cfg.MODEL.VID.MEGA.MAX_OFFSET)
        ref_id = frame_ids_list[ref_relative_id]
        ref_filename = self.pattern[idx] % ref_id
        img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
        target_ref = self.get_groundtruth(self.image_set_index.index(ref_filename))
        img_refs_l.append((img_ref,target_ref))

        img_refs_g = []
        if self.cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:  # 
            size = self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == start_frame_id else 1  # GLOBAL.SIZE == 25
            shuffled_index = self.shuffled_index[str(self.start_id[idx])]
            # shuffled_index = self.shuffled_index[video_name]  # 这样写可读性更强， 与上面等价
            for id in range(size):
                temp = (idx - self.start_id[idx] + self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) 
                # 这个 temp 是相对位置
                # 既然是从shuffled_index中取出来，为什么还要搞的temp 按照顺序呢？
                # 猜测： 这个 temp 是为了把 frame_id==0 和 frame_id >0 区分开， 同时也是确保每一帧取的 temp值不一样
                # frame_id == 0时： size == GLOBAL.SIZE
                # idx - self.start_id[idx] ==0,  id 从 0~ size -1 ， 那 temp= GLOBAL.SIZE - id -1 就从 GLOBAL.SIZE-1 变化到0
                # frame_id > 0 时， size == 1, 这里的for 循环只执行一次， id=0, GLOBAL.SIZE - id - 1 == GLOBAL.SIZE  - 1
                # idx - self.start_id[idx] 是相对起始帧的偏移量， (注意：idx - self.start_id[idx] 不一定等于 frame_id， 对于有些帧没有annotation的被过滤掉的就不等于)
                # 然后 temp 就从 GLOBAL.SIZE  - 1 变化到  GLOBAL.SIZE  - 1 + total_frames, 其中 total_frames 是过滤之后的总帧数
                # 这样一来，就把起始帧和非起始帧取的 global memory 区分开了，同时也把每一帧的 global_memory区分开了

                # print(temp,len(shuffled_index))
                temp_mod = temp % len(frame_ids_list) # 取 mod
                

                filename_g = self.pattern[idx] % shuffled_index[temp_mod]
                img_ref = Image.open(self._img_dir % filename_g).convert("RGB")
                target_ref = self.get_groundtruth(self.image_set_index.index(filename_g))
                #Problem: 如果 filename_g 不在 self.image_set_index 中怎么办, i.e, 它被 filter_annotation 过滤掉了， 怎么办？
                #因为在原来的代码中， 是不用获取 filename_g 的 target的
                # 先看一下 self.annos 里每个 video 被过滤的情况
                img_refs_g.append((img_ref,target_ref))

        target = self.get_groundtruth(idx)  # target is BoxList
        # for box in target.bbox:
        #     print(box.to(torch.int64).tolist(),"bef",target)

        target = target.clip_to_image(remove_empty=True)
        # for box in target.bbox:
        #     print(box.to(torch.int64).tolist(),"aft",target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)  # 对于 target的 transform， 是根据对图片的resize来做target的resize
            for i in range(len(img_refs_l)):
                img_refs_l[i] = self.transforms(img_refs_l[i][0], img_refs_l[i][1])
            for i in range(len(img_refs_g)):
                img_refs_g[i] = self.transforms(img_refs_g[i][0], img_refs_g[i][1])
        else:
            assert False, "should use transforms"

        images = {}
        images["cur"] = (img,target)    # img 是tensor ， transforms 的最后有to_tensor 的操作
        images["ref_l"] = img_refs_l    # list[tuple], each one is (img_ref,target_ref)
        images["ref_g"] = img_refs_g    # list[tuple], each one is (img_g,target_g)
        images["frame_category"] = frame_category
        images["seg_len"] = self.frame_seg_len[idx]
        images["pattern"] = self.pattern[idx]       # # e.g., 0000_2401075277/%06d
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms
        images["filename"] = filename

        return images, idx


    def __len__(self):  # maintain 
        return len(self.image_set_index)


    def filter_annotation(self):  # 已改好
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
        tids = []
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
            gt_classes.append(self.Name2Id[gt_class])
            tids.append(obj["tid"])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # shape == (num_box,)
        tids = torch.tensor(tids)  # shape == (num_box,)
        # print(boxes_tensor.to(torch.int64),"in dataset")
        res = {
            "boxes": boxes_tensor,      #MARK
            "labels": torch.tensor(gt_classes),
            "tids":tids,
            "im_info": im_info,
        }
        return res

    def load_annos(self, cache_file):
        if os.path.exists(cache_file):
            print("loading annotation information from {} ... ".format(cache_file))
            with open(cache_file, "rb") as fid:
                annos = pickle.load(fid)
            print("Done.")
            return annos
        

        annos = []
        outer_idx = 0
        print("construct annos... ")
        for video_name in tqdm(self.video_name_list):
            with open(self._anno_path % tuple(video_name.split('_')),'r') as json_file: 
                video_ann = json.load(json_file)
            width_height = (int(video_ann["width"]),int(video_ann["height"]))
            frame_count = self.video_frame_count[video_name]
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
        cache_dir = os.path.join(self.data_dir, 'VidOR_cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir


    def get_groundtruth(self, idx):
        anno = self.annos[idx]

        width, height = anno["im_info"]  # NOTE im_info is w,h
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"]) 
        target.add_field("tids", anno["tids"])

        return target



def extract_VidOR_gt_features(part_id,gpu_id):
    from mega_core.config import cfg
    from mega_core.data.transforms.build import build_transforms
    from mega_core.utils.checkpoint import DetectronCheckpointer

    DIM_FEAT = 1024
    save_dir = "mega_boxfeatures/GT_boxfeatures/VidORtrain_freq1_part{:02d}".format(part_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    config_file = "configs/MEGA/partxx/VidORtrain_freq1_part{:02d}.yaml".format(part_id)
    BASE_CONFIG = "configs/BASE_RCNN_1gpu.yaml"
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(config_file)

    exclude_video_list = sorted(os.listdir(save_dir))
    exclude_video_list = [v[:-4] for v in exclude_video_list] # "0000_2401075277.npy" --> "0000_2401075277"
    
    dataset = VidORDatasetGt(
        cfg = cfg,
        image_set = "VidORtrain_freq1_part{:02d}".format(part_id),
        img_index = "datasets/vidor-dataset/img_index/VidORtrain_freq1_part{:02d}.txt".format(part_id),
        data_dir = "datasets",
        img_dir = "datasets/vidor-dataset/train_frames",
        anno_path = "datasets/vidor-dataset/annotation/training",
        transforms = build_transforms(cfg,is_train=False),
        exclude_video_list = exclude_video_list
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        drop_last = False,
        shuffle = False,
        collate_fn = lambda x : x 
    )
    

    device = torch.device("cuda:{}".format(gpu_id))
    model = FeatureExtractor(cfg,dataset.image_set_index.copy(),dataset.annos,dataset.filtered_frame_idx)
    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT, use_latest=False, flownet=None)
    model = model.to(device)
    # print(model.device)
    model.eval()

    
    print("start extract features...")
    video_name_list = [x.split('/')[0] for x in dataset.image_set_index].copy()
    video_name_unique = list(set(video_name_list))
    video_name_unique.sort(key=video_name_list.index) # keep the original order
    video_frame_count = {x:video_name_list.count(x) for x in video_name_unique}
    video_tracklets = defaultdict(list)
    for data in tqdm(dataloader):
        images, idx = data[0]
        if images == None:
            continue
        video_name,frame_id = images["filename"].split("/")  # e.g., "0000_2401075277/000010"
        save_path = os.path.join(save_dir,video_name+".npy")

        frame_id = int(frame_id)
        img,target = images["cur"]
        xywh = target.convert('xywh').bbox.cpu().numpy()  # shape == (num_box,4)
        tids = target.get_field("tids").cpu().numpy()
        category_id = target.get_field("labels").cpu().numpy()
        num_box = xywh.shape[0]

        box_info = np.zeros(shape=(num_box,12+DIM_FEAT))
        box_info[:,0] = frame_id          # frame_id        
        box_info[:,1] = tids              # tracklet id, Not necessarily continuous
        box_info[:,2:6] = xywh            # tracklet xywh predicted by deepSORT algorithm (for gt, same as box_info[8:12])
        box_info[:,6] = 1.0               # confidence score, 1.0 for gt
        box_info[:,7] = category_id       # object category id obtained from the detector (MEGA)
        box_info[:,8:12] =  xywh          # object bbox from the detector (MEGA)
        # box_info[:,12:] = box_features  # box appearance feature
        
        
        images["cur"] = (img.to(device),target.to(device))

        for key in ("ref", "ref_l", "ref_m", "ref_g"):
            if key in images.keys():
                images[key] = [(img.to(device),target.to(device)) for img,target in images[key]]
        
        with torch.no_grad():
            box_features = model(images)  # shape == (num_box,1024)
        box_info[:,12:] = box_features.cpu().numpy()

        video_tracklets[video_name].append(box_info)
        video_frame_count[video_name] -= 1
        if video_frame_count[video_name] == 0:
            box_info_per_video =  video_tracklets[video_name]
            box_info_per_video = np.concatenate(box_info_per_video,axis=0)  # shape == (NUM_box,1024)
            np.save(save_path,box_info_per_video)

            video_tracklets.pop(video_name)


def test_dataset(part_id):
    from mega_core.config import cfg
    from mega_core.data.transforms.build import build_transforms

    DIM_FEAT = 1024
    save_dir = "mega_boxfeatures/GT_boxfeatures/VidORtrain_freq1_part{:02d}".format(part_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_file = "configs/MEGA/partxx/VidORtrain_freq1_part{:02d}.yaml".format(part_id)
    BASE_CONFIG = "configs/BASE_RCNN_1gpu.yaml"
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(config_file)
    
    dataset = VidORDatasetGt(
        cfg = cfg,
        image_set = "VidORtrain_freq1_part{:02d}".format(part_id),
        img_index = "datasets/vidor-dataset/img_index/VidORtrain_freq1_part{:02d}.txt".format(part_id),
        data_dir = "datasets",
        img_dir = "datasets/vidor-dataset/train_frames",
        anno_path = "datasets/vidor-dataset/annotation/training",
        transforms = build_transforms(cfg,is_train=False)
    )

    video_name_list = [x.split('/')[0] for x in dataset.image_set_index].copy()
    video_name_unique = list(set(video_name_list))
    video_name_unique.sort(key=video_name_list.index) # keep the original order
    video_frame_count = {x:video_name_list.count(x) for x in video_name_unique}

    total_img_ids = list(range(len(dataset)))
    
    for i in tqdm(range(len(dataset))):
        video_name,frame_id = dataset.image_set_index[i].split("/")
        save_path = os.path.join(save_dir,video_name+".npy")
        if os.path.exists(save_path):
            total_img_ids.pop(i)

    for i in tqdm(total_img_ids):
        video_name,frame_id = dataset.image_set_index[i].split("/")
        frame_id = int(frame_id)
        data = dataset[i]
        images, idx = data
        print(images["cur"][1])

def test_dataloader(part_id):
    from mega_core.config import cfg
    from mega_core.data.transforms.build import build_transforms

    DIM_FEAT = 1024

    config_file = "configs/MEGA/partxx/VidORtrain_freq1_part{:02d}.yaml".format(part_id)
    BASE_CONFIG = "configs/BASE_RCNN_1gpu.yaml"
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(config_file)
    
    dataset = VidORDatasetGt(
        cfg = cfg,
        image_set = "VidORtrain_freq1_part{:02d}".format(part_id),
        img_index = "datasets/vidor-dataset/img_index/VidORtrain_freq1_part{:02d}.txt".format(part_id),
        data_dir = "datasets",
        img_dir = "datasets/vidor-dataset/train_frames",
        anno_path = "datasets/vidor-dataset/annotation/training",
        transforms = build_transforms(cfg,is_train=False)
    )
    

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        drop_last = False,
        shuffle = False,
        collate_fn = lambda x : x 
    )

    for data in tqdm(dataloader):
        # data is a list, len(data) == batch_size
        print(type(data),len(data))
        images, idx = data[0]
        video_name,frame_id = images["filename"].split("/")  # e.g., "0000_2401075277/000010"
        frame_id = int(frame_id)
        
        img,target = images["cur"]
        print(target.bbox)
        xywh = target.convert('xywh').bbox.cpu().numpy()  # shape == (num_box,4)
        tids = target.get_field("tids").cpu().numpy()
        category_id = target.get_field("labels").cpu().numpy()
        num_box = xywh.shape[0]


        box_info = np.zeros(shape=(num_box,12+DIM_FEAT))
        box_info[:,0] = frame_id          # frame_id        
        box_info[:,1] = tids               # tracklet id, Not necessarily continuous
        box_info[:,2:6] = xywh            # tracklet xywh predicted by deepSORT algorithm (for gt, same as box_info[8:12])
        box_info[:,6] = 1.0               # confidence score, 1.0 for gt
        box_info[:,7] = category_id       # object category id obtained from the detector (MEGA)
        box_info[:,8:12] =  xywh          # object bbox from the detector (MEGA)

        print(box_info[:,:12],box_info.shape)
        break

def test_model_weight(part_id):
    from mega_core.config import cfg
    from mega_core.data.transforms.build import build_transforms
    from mega_core.utils.checkpoint import DetectronCheckpointer

    DIM_FEAT = 1024

    config_file = "configs/MEGA/partxx/VidORtrain_freq1_part{:02d}.yaml".format(part_id)
    BASE_CONFIG = "configs/BASE_RCNN_1gpu.yaml"
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(config_file)
    
    dataset = VidORDatasetGt(
        cfg = cfg,
        image_set = "VidORtrain_freq1_part{:02d}".format(part_id),
        img_index = "datasets/vidor-dataset/img_index/VidORtrain_freq1_part{:02d}.txt".format(part_id),
        data_dir = "datasets",
        img_dir = "datasets/vidor-dataset/train_frames",
        anno_path = "datasets/vidor-dataset/annotation/training",
        transforms = build_transforms(cfg,is_train=False)
    )

    model = FeatureExtractor(cfg,dataset.image_set_index.copy(),dataset.annos)
    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT, use_latest=False, flownet=None)

def invesgate_anno(part_id):
    from mega_core.config import cfg
    from mega_core.data.transforms.build import build_transforms
    
    img_index = "datasets/vidor-dataset/img_index/VidORtrain_freq1_part{:02d}.txt".format(part_id)
    with open(img_index) as f:
        lines = [x.strip().split(" ") for x in f.readlines()]  # .strip() 去除字符串首尾的空格

    image_set_index = ["%s/%06d" % (x[0], int(x[1])) for x in lines] # e.g., 0000_2401075277/000010   # 代表每张图片的名字
    pattern = [x[0] + "/%06d" for x in lines]                        # e.g., 0000_2401075277/%06d    # 代表每个video中图片名字的pattern
    frame_seg_id = [int(x[1]) for x in lines]
    frame_seg_len = [int(x[2]) for x in lines]

    video_name_list_ori = [x.split('/')[0] for x in image_set_index].copy()
    video_frame_count_ori = {k:v for k,v in zip(video_name_list_ori,frame_seg_len)}
    

    DIM_FEAT = 1024

    config_file = "configs/MEGA/partxx/VidORtrain_freq1_part{:02d}.yaml".format(part_id)
    BASE_CONFIG = "configs/BASE_RCNN_1gpu.yaml"
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(config_file)
    
    dataset = VidORDatasetGt(
        cfg = cfg,
        image_set = "VidORtrain_freq1_part{:02d}".format(part_id),
        img_index = "datasets/vidor-dataset/img_index/VidORtrain_freq1_part{:02d}.txt".format(part_id),
        data_dir = "datasets",
        img_dir = "datasets/vidor-dataset/train_frames",
        anno_path = "datasets/vidor-dataset/annotation/training",
        transforms = build_transforms(cfg,is_train=False)
    )
    
    # after filter 
    video_name_list = [x.split('/')[0] for x in dataset.image_set_index].copy()
    video_name_unique = list(set(video_name_list))
    video_name_unique.sort(key=video_name_list.index) # keep the original order
    video_frame_count = {x:video_name_list.count(x) for x in video_name_unique}

    xx = {
        "ori":image_set_index.copy(),
        "filtered":dataset.image_set_index.copy()
    }

    import pickle
    with open("test_gramma/video_name_list.pkl",'wb') as f:
        pickle.dump(xx,f)
    print("saved")

    frame_ct1 = []
    frame_ct2 = []

    for video_name in video_name_unique:

        # ValueError: '0000_2440175990/000696' is not in list
        ct1 = video_frame_count_ori[video_name]
        ct2 = video_frame_count[video_name]
        # print(video_name,ct1,ct2)
        frame_ct1.append(ct1)
        frame_ct2.append(ct2)
    
    frame_ct1 = np.array(frame_ct1,dtype=np.int)
    frame_ct2 = np.array(frame_ct2,dtype=np.int)

    num_eq = np.sum(frame_ct1 == frame_ct2)
    print(num_eq,len(frame_ct1)-num_eq)

    
    



if __name__ == "__main__":
    # test_dataloader(part_id=1)
    # test_model_weight(part_id=1)
    
    # invesgate_anno(part_id=1)
    # CUDA_VISIBLE_DEVICES=1 python utils/utils_class_func.py

    # test_dataset(1)

    parser = argparse.ArgumentParser(description='xxxx')
    parser.add_argument('--part_id', type=str, help='the dataset name for evaluation')
    parser.add_argument('--gpu_id', type=str, help='the dataset name for evaluation')
    args = parser.parse_args()

    part_id = int(args.part_id)
    gpu_id = int(args.gpu_id)

    extract_VidOR_gt_features(part_id,gpu_id)
    # python tools/extract_gt_boxfeatures.py --part_id 8 --gpu_id 1
    # 10.214.223.101 (kgl-6):
    # part1cuda3
    # part8cuda2
    

    # 10.214.223.109 (kgl-3):
    # part2cuda0
    # part3cuda1
    # part4cuda2
    # part5cuda3
    # part6cuda0
    # part7cuda0