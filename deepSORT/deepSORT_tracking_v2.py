import os
import pickle
import numpy as np
import random
import cv2
from tqdm import tqdm
from deep_sort_app_v2 import run

### default args:
nms_max_overlap = 1.0
min_detection_height = 0
max_cosine_distance = 0.2
nn_budget = None
display = False
frameRate = 25

### modified args:
min_confidence = 0.3
max_miss = 60


all_frames_path = None
proposal_result_path = "MEGA/mega_boxfeatures/VidORval_freq1.pkl"   # original frame-level proposal results   

save_dir = "deepSORT/tracking_results/VidORval_freq1_m{}s{:.1f}/".format(max_miss,min_confidence)  

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
print("save_dir==",save_dir)

with open(proposal_result_path,'rb') as f:
    proposal_results = pickle.load(f)

video_name_list = list(proposal_results.keys())


for video_name in tqdm(video_name_list):  
    proposals_list = proposal_results[video_name]
    output_file = os.path.join(save_dir, video_name+".npy")
    if os.path.exists(output_file):
        continue
    ## step1 convert to deepSORT_format, refer to original script 'generate_deep_sort_format_v2.py'
    proposals_per_video = []
    for frame_id, proposals in enumerate(proposals_list):
        assert video_name == proposals["video_name"]
        assert frame_id == proposals["frame_id"], "frame_id:{} != {}".format(frame_id,proposals["frame_id"])
        bboxes = proposals["bboxes"]                # refer to  `MEGA/mega_boxfeatures/cvt_proposal_result.py`
        roifeats = proposals["roifeats"]
        scores = proposals["scores"].tolist()
        labels = proposals["labels"].tolist()
        width,height = proposals["width_height"]
        
        for i, bbox in enumerate(bboxes):
            vector_per_box = []
            vector_per_box += [frame_id,-1]
            vector_per_box += bbox.tolist()           # format="xywh"
            vector_per_box += [scores[i]]
            vector_per_box += [labels[i]]
            vector_per_box += [-1,-1]
            vector_per_box += roifeats[i,:].tolist()
            proposals_per_video.append(vector_per_box)
    
    proposals_per_video = np.array(proposals_per_video)

    ## step2. gather_sequence_info, refer to the original function in `gather_sequence_info` in deep_sort_app_v2.py
    if all_frames_path != None:
        video_frames_dir = os.path.join(all_frames_path,video_name)  # .JPEG files for all the frames of the video
        image_filenames = {
            int(f.split(".")[0]): os.path.join(video_frames_dir, f)
            for f in os.listdir(video_frames_dir)}
    else:
        image_filenames = None
    detections = proposals_per_video
    image_size = (height,width)
    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    feature_dim = detections.shape[1] - 10 if detections is not None else 0   
    seq_info = {
        # "sequence_name": os.path.basename(video_frames_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": None,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": 1000 / int(frameRate)
    }
    ## step3. run tracking, use the function `run` from deep_sort_app_v2.py
    run(
        sequence_dir=None,
        detection_file=None,
        min_confidence = min_confidence,
        output_file = output_file,
        nms_max_overlap = nms_max_overlap,
        min_detection_height= min_detection_height,
        max_cosine_distance = max_cosine_distance,
        nn_budget = nn_budget,
        display = display,
        max_miss = max_miss,
        seq_info=seq_info
    )


   


