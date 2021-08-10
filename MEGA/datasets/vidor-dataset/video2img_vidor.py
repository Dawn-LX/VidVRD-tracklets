import os
import cv2
from tqdm import tqdm

def sorted_listdir(dir):
    ls = os.listdir(dir)
    return sorted(ls)

val_videos_dir = "MEGA/datasets/vidor-dataset/val_videos/"
val_frames_dir = "MEGA/datasets/vidor-dataset/val_frames/"

video_groups = sorted_listdir(val_videos_dir)
for video_group in tqdm(video_groups):
    video_names = sorted_listdir(os.path.join(val_videos_dir,video_group))
    for video_name in video_names:
        video_path = os.path.join(val_videos_dir,video_group,video_name)
        video_name = video_group + "_" + video_name.split('.')[0]
        # print(video_name)
        frames_dir = os.path.join(val_frames_dir,video_name)
        if not os.path.exists(frames_dir):
            os.mkdir(frames_dir)
        
        cap = cv2.VideoCapture(video_path)  
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success = True
        count = 0
        while success and count < n_frames:
            success, image = cap.read()
            if success:
                cv2.imwrite(os.path.join(frames_dir,"{:06d}.JPEG".format(count)), image)
                count+=1



