import os
import json
import numpy as np
import torch

from tqdm import tqdm
from evaluation import eval_video_object
from utils.categories_v2 import vidor_CatId2name,vidor_categories

DIM_FEATURE = 1024   # dimension of visual appearance feature (RoI pooled feature), this is determined by MEGA



def prepare_gt(annotation_dir,category_map):

    group_ids = sorted(os.listdir(annotation_dir))
    video_ann_paths = []
    for gid in group_ids:
        group_dir = os.path.join(annotation_dir,gid)
        ann_names = sorted(os.listdir(group_dir))
        paths = [os.path.join(group_dir,ann_name) for ann_name in ann_names]
        video_ann_paths += paths

    gt_results = {}
    for video_ann_path in tqdm(video_ann_paths):
        temp = video_ann_path.split('/')[-2:]
        video_name = temp[0] + '_' + temp[1].split('.')[0]
        
        with open(video_ann_path,'r') as f:
            video_anno = json.load(f)


        video_len = len(video_anno["trajectories"])
        video_wh = (video_anno["width"],video_anno["height"])

        traj_categories = video_anno["subject/objects"]      
        # tid2category_map = [traj["category"] for traj in traj_categories] #  this is WRONG, tids do not necessarily cover 0 ~ len(traj_categories)-1
        tid2category_map = {traj["tid"]:traj["category"] for traj in traj_categories} # we need this
        # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'}
        trajs = {traj["tid"]:{} for traj in traj_categories}

        for tid in trajs.keys():
            trajs[tid]["all_bboxes"] = []
            trajs[tid]["frame_ids"] = []
            trajs[tid]["category"] = tid2category_map[tid]
        
        for frame_id,frame_anno in enumerate(video_anno["trajectories"]):
            for bbox_anno in frame_anno:
                tid = bbox_anno["tid"]
                bbox = bbox_anno["bbox"]
                bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
                trajs[tid]["all_bboxes"].append(bbox)
                trajs[tid]["frame_ids"].append(frame_id)

        result_per_video = []
        for tid in trajs.keys():
            all_bboxes = trajs[tid]["all_bboxes"]
            frame_ids = trajs[tid]["frame_ids"]
            category = category_map[trajs[tid]["category"]]
            result_per_video.append(
                {
                    "category":category,
                    "trajectory":{fid:box for fid,box in zip(frame_ids,all_bboxes)}
                }
            )
        
        gt_results.update({video_name:result_per_video})

    return gt_results

def prepare_tracking_results(tracking_results_dir,min_frames_th,max_proposal,score_th,category_map):
    tracking_results = {}

    tracklets_list = sorted(os.listdir(tracking_results_dir))
    for filename in tqdm(tracklets_list):
        video_name = filename.split('.')[0]
        track_res_path = os.path.join(tracking_results_dir,filename)
        track_res = np.load(track_res_path,allow_pickle=True)

        trajs = {box_info[1]:{} for box_info in track_res}
        for tid in trajs.keys():  
            trajs[tid]["frame_ids"] = []
            trajs[tid]["all_bboxes"] = []
            trajs[tid]["scores"] = []
            trajs[tid]["category_id"] = []   

        for idx,box_info in enumerate(track_res):
            if not isinstance(box_info,list):
                box_info = box_info.tolist()
            assert len(box_info) == 6 or len(box_info) == 12 + DIM_FEATURE,"len(box_info)=={}".format(len(box_info))
            
            frame_id = box_info[0]
            tid = box_info[1]
            tracklet_xywh = box_info[2:6]
            xmin_t,ymin_t,w_t,h_t = tracklet_xywh
            xmax_t = xmin_t + w_t
            ymax_t = ymin_t + h_t
            bbox_t = [xmin_t,ymin_t,xmax_t,ymax_t]
            confidence = float(0)
            if len(box_info) == 12 + DIM_FEATURE:
                confidence = box_info[6]
                cat_id = box_info[7]
                xywh = box_info[8:12]
                xmin,ymin,w,h = xywh
                xmax = xmin+w
                ymax = ymin+h
                bbox = [(xmin+xmin_t)/2, (ymin+ymin_t)/2, (xmax+xmax_t)/2,(ymax+ymax_t)/2]
                # roi_feature = box_info[12:]
                trajs[tid]["category_id"].append(cat_id)
                
            if len(box_info) == 6:
                trajs[tid]["all_bboxes"].append(bbox_t)
            else:
                trajs[tid]["all_bboxes"].append(bbox)
            trajs[tid]["frame_ids"].append(frame_id)    
            trajs[tid]["scores"].append(confidence)

        cat_ids = []
        bboxes_list = []
        frame_ids_list = []
        scores = []
        for tid in trajs.keys():
            if trajs[tid]["category_id"] == []:
                category_id = 0
            else:
                category_id = np.argmax(np.bincount(trajs[tid]["category_id"]))  # 求众数
            
            if len(trajs[tid]["frame_ids"]) < min_frames_th or category_id == 0:
                continue
            
            score = np.mean(trajs[tid]["scores"])

            cat_ids.append(category_id)
            scores.append(score)
            bboxes_list.append(trajs[tid]["all_bboxes"])
            frame_ids_list.append(trajs[tid]["frame_ids"])
            


        # score_clipping:
        cat_ids = torch.tensor(cat_ids)
        scores = torch.tensor(scores)
        index = torch.where(scores > score_th)[0]
        # assert len(index) > 0
        if len(index) == 0:
            print("video:{} has no proposal after score clipping (score_th={:.2f})".format(video_name,score_th))
            proposal_results.update({video_name:[]})
            continue


        scores = scores[index]
        bboxes_list = [bboxes_list[ii] for ii in index]   # format: xyxy
        frame_ids_list = [frame_ids_list[ii] for ii in index]
        cat_ids = cat_ids[index]

        # proposal num clipping
        index = torch.argsort(scores,descending=True)
        index = index[:max_proposal]
        scores = scores[index].tolist()
        cat_ids = cat_ids[index].tolist()
        bboxes_list = [bboxes_list[ii] for ii in index]
        frame_ids_list = [frame_ids_list[ii] for ii in index]


        result_per_video = []
        for idx,cat_id in enumerate(cat_ids):
            bboxes = bboxes_list[idx]
            frame_ids = frame_ids_list[idx]
            score = scores[idx]
            category = vidor_CatId2name[cat_id]
            category = category_map[category]
            result_per_video.append(
                {
                    "category":category,
                    "score":score,
                    "trajectory":{fid:box for fid,box in zip(frame_ids,bboxes)}
                }
            )
        tracking_results.update({video_name:result_per_video})
    
def reset_category(only_pos):
    if only_pos:
        fgbg_map = {x["name"]:"fg" for x in vidor_categories}
        fgbg_map["__background__"] = "bg"
        return fgbg_map
    else:
        identity_map = {x["name"]:x["name"] for x in vidor_categories}
        return identity_map



if __name__ == "__main__":
    # tracking_results_dir = "deepSORT/tracking_results/VidORval_freq1_m60s0.3"
    tracking_results_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORval_freq1"
    annotation_dir = "/home/gkf/project/deepSORT/datasets/vidor-dataset/annotation/validation"
    
    only_pos=False
    category_map = reset_category(only_pos)

    gt_results = prepare_gt(annotation_dir,category_map)

    
    tracking_results = prepare_tracking_results(
        tracking_results_dir = tracking_results_dir,
        min_frames_th = 15,
        max_proposal = 180,
        score_th = 0.4,
        category_map = category_map
    )

    eval_video_object(gt_results,tracking_results)




## ----------- MEGA inference freq1 -----------
# deepSORT/tracking_results/VidORval_freq1_m60s0.3 mean AP 0.1248

# /home/gkf/project/vidvrd-mff/ORval_traj.pkl  mean AP  0.0882
# /home/gkf/project/vidvrd-mff/ORval_traj_FG.pkl (only consider bbox postion) mean AP 0.1664   

