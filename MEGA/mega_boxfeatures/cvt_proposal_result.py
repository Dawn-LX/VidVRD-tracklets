import os
import torch
import pickle
import numpy
import json
from tqdm import tqdm

video_ann_path = "datasets/vidor-dataset/annotation/validation"
predictions_path = "training_dir/COCO34ORfreq32_4gpu/inference_180k/inference/VidORval_freq1/predictions.pth"
save_path = "mega_boxfeatures/VidORval_freq1.pkl"
print("save_path==",save_path)
# assert os.path.exists(save_path)
print("loading predictions...")
predictions = torch.load(predictions_path)
print("predictions loaded")
img_index_path = "datasets/vidor-dataset/img_index/VidORval_freq1.txt"


print("constructing img_index....")
with open(img_index_path,'r') as f:
    img_index = f.readlines()
img_index = [item.strip().split(" ") for item in img_index]
print("img_index has been constructed")
# print(img_index[:12])
print(type(predictions),len(predictions))
print(predictions[0])
boxlist = predictions[0]
scores = boxlist.get_field("scores")
print(boxlist.fields())

# print(scores)
roifeats = boxlist.get_field("roi_feats")
# torch.set_printoptions(profile="full")
# print(roifeats)
# print(roifeats[0].shape,"roi_feats.shape--------")

print("converting predictions to proposal format....")
results_with_RoI = {}
for idx, boxlist in enumerate(predictions):
    video_name = img_index[idx][0]
    results_with_RoI[video_name] = []

for idx, boxlist in enumerate(tqdm(predictions)): 
    video_name = img_index[idx][0]
    frame_id = int(img_index[idx][1])
    # print(boxlist)
    temp = video_name.split("_")
    video_ann = os.path.join(video_ann_path,temp[0],temp[1] + '.json')
    with open(video_ann,'r') as f:
        video_ann = json.load(f)
    
    assert temp[1] == video_ann["video_id"]
    width_height = (video_ann["width"],video_ann["height"])
    boxlist = boxlist.resize(width_height)
    boxlist = boxlist.convert('xywh')
    bboxes = boxlist.bbox.numpy()
    scores = boxlist.get_field("scores").numpy()
    labels = boxlist.get_field("labels").numpy()
    roifeats = boxlist.get_field("roi_feats").numpy()
    frame_proposal_dict = {
        "video_name":video_name,
        "frame_id":frame_id,
        "width_height":width_height,
        "bboxes":bboxes,
        "scores":scores,
        "labels":labels,
        "roifeats":roifeats
    }
    results_with_RoI[video_name].append(frame_proposal_dict)


print(boxlist.size)
print(boxlist)
print("saving results....")
with open(save_path,'wb') as f:
    pickle.dump(results_with_RoI,f)

print("-----------finish!-------------")
