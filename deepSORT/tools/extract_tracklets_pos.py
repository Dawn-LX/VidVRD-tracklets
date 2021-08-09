import os
import numpy as np
from tqdm import tqdm

def extract_pos():
    dim_boxfeature = 1024
    load_dir = "tracking_results/miss60_minscore0p3/VidORval_freq1"
    save_dir = "trackletsPosOnly/VidORval_freq1_m60s0.3/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename_list = sorted(os.listdir(load_dir))
    for filename in tqdm(filename_list):
        load_path = os.path.join(load_dir,filename)
        save_path = os.path.join(save_dir,filename)
        track_res = np.load(load_path,allow_pickle=True)
        track_pos = []
        for idx,box_info in enumerate(track_res):
            if not isinstance(box_info,list):
                box_info = box_info.tolist()
            assert len(box_info) == 6 or len(box_info) == 12 + dim_boxfeature,"len(box_info)=={}".format(len(box_info))

            track_pos.append(
                box_info[:12]
            )

            """
            frame_id = box_info[0]
            tid = box_info[1]
            tracklet_xywh = box_info[2:6]
            
            if len(box_info) == 12 + dim_boxfeature:
                confidence = box_info[6]
                cat_id = box_info[7]
                xywh = box_info[8:12]
                roi_feature = box_info[12:]
            """
        track_pos = np.array(track_pos)
        np.save(save_path,track_pos)
        

if __name__ == "__main__":
    extract_pos()

    # xx = np.load("trackletsPosOnly/VidORval_freq1_m60s0.3/0001_2793806282.npy",allow_pickle=True)
    # for x in xx:
    #     print(type(x),len(x))
    # print(xx,type(xx),xx.shape)