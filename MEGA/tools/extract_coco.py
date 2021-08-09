import json
import pickle
from tqdm import tqdm
from pycocotools.coco import COCO
from categories_v2 import vidor_categories

def extract_COCO_data(ann_file,output_file):
    vidor_class80 = [c["name"] for c in vidor_categories]

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

    vidor_coco_intersection = []

    for c in vidor_class80:
        if c in COCO_CATEGORIES:
            vidor_coco_intersection.append(c)

    print("classes that both in vidor and coco: {}, length == {}".format(vidor_coco_intersection,len(vidor_coco_intersection)))
    synonyms_vidor2coco = {
        "sofa":"couch",
        "ball/sports_ball":"sports ball",
        "stop_sign":"stop sign",
        "traffic_light":"traffic light",
        "cattle/cow":"cow",
        "sheep/goat":"sheep"
    }
    vidor_coco_intersection += list(synonyms_vidor2coco.values())
    print("after add Synonyms, length=={}".format(len(vidor_coco_intersection)))



    with open(ann_file,'r') as f:
        ann_json = json.load(f)

    print(ann_json.keys())
    for k in ann_json.keys():
        print(k,type(ann_json[k]),len(ann_json[k]))


    cocoCatId2CatName = {cat["id"]:cat["name"] for cat in ann_json["categories"]}

    imgid_to_imgname_map = {img_info["id"]:img_info["file_name"] for img_info in ann_json["images"]}
    imgid_to_wh_map = {img_info["id"]:[int(img_info["width"]),int(img_info["height"])] for img_info in ann_json["images"]}


    category_to_catId_map = {cat["name"]:cat["id"] for cat in ann_json["categories"]}

    coco = COCO(ann_file)


    vidor_coco_intersection = [category_to_catId_map[cat] for cat in vidor_coco_intersection]
    print(vidor_coco_intersection)

    #  img_ids = coco.getImgIds(catIds=vidor_coco_intersection) # this is WRONG, this return images which contain all the categories in vidor_coco_intersection
    
    # Instead, we use the following for-loop
    # each image contains at least one object with category in vidor_coco_intersection,  but can also contain objects whose categories not in vidor_coco_intersection
    imgids = [] # image ids that contain VidOR catrgories
    for cat_id in vidor_coco_intersection:
        imgids_per_cat = coco.getImgIds(catIds=[cat_id])   
        imgids += imgids_per_cat                           
    

    imgids = list(set(imgids))  # remove duplicate img_ids


    imgid2names = {idx:imgid_to_imgname_map[idx] for idx in imgids}
    imgid2wh = {idx:imgid_to_wh_map[idx] for idx in imgids}
    
    imgid2annos = {}
    for img_id in tqdm(imgids):
        annoids = coco.getAnnIds(imgIds=img_id, iscrowd=False)  # 获取某张图片对应的所有ann_ids
        annos = coco.loadAnns(annoids)
        selected_annos = []
        for ann in annos:
            if ann["category_id"] in vidor_coco_intersection:
                selected_annos.append(ann)
        
        if len(selected_annos) > 0:
            imgid2annos.update({img_id: selected_annos})  

    

    imgid_annos_dict = {
        "imgids":imgids,
        "imgid2names":imgid2names,
        "imgid2wh":imgid2wh,
        "imgid2annos":imgid2annos,
        "cocoCatId2CatName":cocoCatId2CatName,
        "synonyms_vidor2coco":synonyms_vidor2coco,
    }
    with open(output_file,'wb') as f:
        pickle.dump(imgid_annos_dict,f)
        f.close()

if __name__ == "__main__":
    ann_file = "/home/gkf/COCOdataset/annotations/instances_train2014.json"
    output_file = "MEGA/datasets/COCOinVidOR/COCO_train_34classes.pkl"

    # ann_file = "/home/gkf/COCOdataset/annotations/instances_valminusminival2014.json"
    # output_file = "MEGA/datasets/COCOinVidOR/COCO_valmini_34classes.pkl"

    extract_COCO_data(ann_file,output_file)