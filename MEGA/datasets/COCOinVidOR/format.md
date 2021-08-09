
## NOTE

please use `MEGA/tools/extract_coco.py` to generate `COCO_valmini_34classes.pkl` and `COCO_train_34classes.pkl`

```json5
# We pick images which contain VidOR's categories from coco, and exclude all the crowd annotations. 
# COCO_valmini_34classes.pkl: constructed from coco_val2014_minus_minival2014
# COCO_train_34classes.pkl: construcetd from coco_train2014
# format of these .pkl files:
{
    "imgids":imgids,
    "imgid2names":imgid2names,
    "imgid2wh":imgid2wh,
    "imgid2annos":imgid2annos,
    "cocoCatId2CatName":cocoCatId2CatName,
    "synonyms_vidor2coco":synonyms_vidor2coco
}
# imgids is a list, these images contain VidOR categories
# in COCO_train_34classes.pkl : len(imgids) == 58890 
# in COCO_valmini_34classes.pkl: len(imgids)== 25407

# format of imgid2names
{
    131089: "COCO_val2014_000000131089.jpg",  # a fake example
    ...
    xxxxxx: "COCO_val2014_000000xxxxxx.jpg"
}

# format of imgid2wh:
{
    131089: [640,480],  # a fake example
    ...
    xxxxxx: [768,512]
}

# format of imgid2annos:
{
    131089: annos_of_img_131089,   # a fake example
    ...
    xxxxxx: annos_of_img_xxxxxx
}
# annos_of_img_xxxxxx is a list, len == number of objects in img_xxxxxx

# format of annos_of_img_xxxxxx's item is just the same as that in coco
# e,g, an annos_of_img_xxxxxx[i] is like that:
{
    'segmentation': [[35.5, 1.76, 13.43, 16.15, 47.02, 19.03, 113.23, 24.79, 172.72, 47.82, 182.31, 51.66, 205.34, 56.45, 226.45, 56.45, 239.89, 57.41, 202.47, 3.68]], 
    'area': 6085.1964499999995, 
    'iscrowd': 0,                   
    'image_id': 262145, 
    'bbox': [13.43, 1.76, 226.46, 55.65], 
    'category_id': 28, 
    'id': 285569
}
# Note that we have excluded all the crowd annos, 
# i.e., annos_of_img_xxxxxx[i]["iscrowd"] == 0 for each annotation

# cocoCatId2CatName: Note that the category ids in coco are not continuous
cocoCatId2CatName = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# synonyms_vidor2coco:
synonyms_vidor2coco = {
    "sofa":"couch",
    "ball/sports_ball":"sports ball",
    "stop_sign":"stop sign",
    "traffic_light":"traffic light",
    "cattle/cow":"cow",
    "sheep/goat":"sheep"
} # len == 6

# classes that both in VidOR and coco: 
vidor_coco_intersection = [
    'cake', 'backpack', 'handbag', 'laptop', 'suitcase',
    'frisbee', 'skateboard', 'snowboard', 'surfboard',
    'bottle', 'chair', 'cup', 'microwave', 'oven',
    'refrigerator', 'sink', 'toilet', 'bench',
    'bicycle', 'car', 'motorcycle', 'train',
    'bird', 'bear', 'cat', 'dog', 'elephant', 'horse'
] # len(vidor_coco_intersection) == 28

# totoal : 28+6=34

```