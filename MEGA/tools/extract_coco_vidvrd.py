import json
import pickle
from tqdm import tqdm
from pycocotools.coco import COCO
from categories_v2 import vidvrd_categories

vidvrd_class35 = vidvrd_categories

print(vidvrd_class35,len(vidvrd_class35))
COCO_CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

vidvrd_coco_intersection = []

for c in vidvrd_class35:
    if c in COCO_CATEGORIES:
        vidvrd_coco_intersection.append(c)

print("classes that both in vidvrd and coco: {}, length == {}".format(vidvrd_coco_intersection,len(vidvrd_coco_intersection)))


ann_file = "/home/gkf/COCOdataset/annotations/instances_train2014.json"
output_file = "datasets/COCOinvidvrd/COCO_train_16classes.pkl"

# ann_file = "/home/gkf/COCOdataset/annotations/instances_valminusminival2014.json"
# output_file = "datasets/COCOinvidvrd/COCO_valmini_16classes.pkl"

with open(ann_file,'r') as f:
    ann_json = json.load(f)

print(ann_json.keys())
for k in ann_json.keys():
    print(k,type(ann_json[k]),len(ann_json[k]))

# print(ann_json["categories"])
# print(ann_json["images"][1])
# print(ann_json["annotations"][1])

cocoCatId2CatName = {cat["id"]:cat["name"] for cat in ann_json["categories"]}
print(cocoCatId2CatName)
imgid_to_imgname_map = {img_info["id"]:img_info["file_name"] for img_info in ann_json["images"]}
imgid_to_wh_map = {img_info["id"]:[int(img_info["width"]),int(img_info["height"])] for img_info in ann_json["images"]}

print(len(imgid_to_imgname_map),"imgid_to_imgname_map------------")
category_to_catId_map = {cat["name"]:cat["id"] for cat in ann_json["categories"]}

coco = COCO(ann_file)


vidvrd_coco_intersection = [category_to_catId_map[cat] for cat in vidvrd_coco_intersection]
print(vidvrd_coco_intersection)

#  img_ids = coco.getImgIds(catIds=vidvrd_coco_intersection) 
# 注意：这个catIds是说同时包含这几种categories的图片，求交集, 所以不能直接这样用
# 要用下面那个循环
imgids = [] # image ids that contain vidvrd catrgories
for cat_id in vidvrd_coco_intersection:
    imgids_per_cat = coco.getImgIds(catIds=[cat_id])   # 这些图片虽然是包含 vidvrd中的类的，但会出现一种情况：某张图片同时包含一个在vidvrd中的class和一个不在vidvrd中的class
    imgids += imgids_per_cat                           # i.e., 每张图片中至少有一个object是属于vidvrd的类的，但也会存在一些object不属于vidvrd的类

print(len(imgids))
imgids = list(set(imgids))
print("after list(set(..)): len==",len(imgids))  # len==24611

imgid2names = {idx:imgid_to_imgname_map[idx] for idx in imgids}
imgid2wh = {idx:imgid_to_wh_map[idx] for idx in imgids}
print(len(imgid2names),"imgid2names------------")
print(imgid2names[imgids[2]],imgids[2])
imgid2annos = {}
for img_id in tqdm(imgids):
    annoids = coco.getAnnIds(imgIds=img_id, iscrowd=False)  # 获取某张图片对应的所有ann_ids
    annos = coco.loadAnns(annoids)
    selected_annos = []
    for ann in annos:
        if ann["category_id"] in vidvrd_coco_intersection:
            selected_annos.append(ann)
    
    if len(selected_annos) > 0:
        imgid2annos.update({img_id: selected_annos})  
        if img_id == 262148:
            print(len(selected_annos),"---------------len selected") 

print(len(imgid2annos))

imgid_annos_dict = {
    "imgids":imgids,
    "imgid2names":imgid2names,
    "imgid2wh":imgid2wh,
    "imgid2annos":imgid2annos,
    "cocoCatId2CatName":cocoCatId2CatName,
}
with open(output_file,'wb') as f:
    pickle.dump(imgid_annos_dict,f)
    f.close()

