import os
import json
from tqdm import tqdm
img_path = "MEGA/datasets/vidor-dataset/val_frames"
ann_dir = "MEGA/datasets/vidor-dataset/annotation/validation"
freq = 1
save_file = "MEGA/datasets/vidor-dataset/img_index/VidORval_freq"+ str(freq) +".txt"
video_name_list = os.listdir(img_path)
video_name_list = sorted(video_name_list)
print(len(video_name_list))
to_be_written = []
for video_name in tqdm(video_name_list):  # 这里的 video_name 包括了 vidvrd的train和test的数据
    group_id,video_id = video_name.split("_")
    train_video_ann_path = os.path.join(ann_dir,group_id,video_id+'.json')
    with open(train_video_ann_path,'r') as f:
        video_anno = json.load(f)
    
    frames = os.listdir(os.path.join(img_path,video_name))
    video_len = len(frames)
    frame_anno_len = len(video_anno["trajectories"])
    # assert video_len == video_anno["frame_count"]  # 对于 vidor，有些不成立这个， video_len =  video_anno["frame_count"]-1
    assert video_len == video_anno["frame_count"] or video_len == video_anno["frame_count"]-1
    assert frame_anno_len == video_anno["frame_count"]
    
    for frame_id in range(0,video_len,freq):
        index_map = video_name + ' ' + str(frame_id) + ' ' + str(video_len) + ' ' + str(frame_anno_len) +  '\n'
        to_be_written.append(index_map)



to_be_written = sorted(to_be_written, key=lambda x: x.split(' ')[0])  # 按vodeo的名字排序

with open(save_file,'w') as f:
    f.writelines(to_be_written)