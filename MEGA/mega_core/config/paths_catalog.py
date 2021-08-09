# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy

class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        ##############################################
        "DET_train_30classes": {
            "img_dir": "ILSVRC2015/Data/DET",
            "anno_path": "ILSVRC2015/Annotations/DET",
            "img_index": "ILSVRC2015/ImageSets/DET_train_30classes.txt"
        },
        "VID_train_15frames": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_train_15frames.txt"
        },
        "VID_train_every10frames": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_train_every10frames.txt"
        },
        "VID_val_frames": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_val_frames.txt"
        },
        "VID_val_videos": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_val_videos.txt"
        },
        "COCO_valmini_34classes":
        {
            "img_dir": "COCOdataset/val2014",
            "anno_path": "COCOinVidOR/COCO_valmini_34classes.pkl"
        },
        "COCO_train_34classes":
        {
            "img_dir": "COCOdataset/train2014",
            "anno_path": "COCOinVidOR/COCO_train_34classes.pkl"
        },
        "VidORtrain_freq32":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq32.txt"
        },
        "VidORval_freq32":
        {
            "img_dir": "vidor-dataset/val_frames",
            "anno_path": "vidor-dataset/annotation/validation",
            "img_index": "vidor-dataset/img_index/VidORval_freq32.txt"
        },
        #### --------------- inference  freq1----------------------------
        "VidORval_freq1":
        {
            "img_dir": "vidor-dataset/val_frames",
            "anno_path": "vidor-dataset/annotation/validation",
            "img_index": "vidor-dataset/img_index/VidORval_freq1.txt"
        },
        "VidORval_freq1_0024":  #
        {
            "img_dir": "vidor-dataset/val_frames",
            "anno_path": "vidor-dataset/annotation/validation",
            "img_index": "vidor-dataset/img_index/VidORval_freq1_0024.txt"
        },
        "VidORval_freq2_0024":  #
        {
            "img_dir": "vidor-dataset/val_frames",
            "anno_path": "vidor-dataset/annotation/validation",
            "img_index": "vidor-dataset/img_index/VidORval_freq2_0024.txt"
        },
        "VidORtrain_freq1_0_999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq1_0_999.txt"
        },
        "VidORtrain_freq1_1k1999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq1_1k1999.txt"
        },
        "VidORtrain_freq1_2k2999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq1_2k2999.txt"
        },
        "VidORtrain_freq1_3k3999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq1_3k3999.txt"
        },
        "VidORtrain_freq1_4k4999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq1_4k4999.txt"
        },
        "VidORtrain_freq1_5k5999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq1_5k5999.txt"
        },
        "VidORtrain_freq1_6k6999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq1_6k6999.txt"
        },
        #### --------------- inference  freq2----------------------------
        "VidORval_freq2":
        {
            "img_dir": "vidor-dataset/val_frames",
            "anno_path": "vidor-dataset/annotation/validation",
            "img_index": "vidor-dataset/img_index/VidORval_freq2.txt"
        },
        "VidORtrain_freq2_0_999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq2_0_999.txt"
        },
        "VidORtrain_freq2_1k1999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq2_1k1999.txt"
        },
        "VidORtrain_freq2_2k2999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq2_2k2999.txt"
        },
        "VidORtrain_freq2_3k3999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq2_3k3999.txt"
        },
        "VidORtrain_freq2_4k4999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq2_4k4999.txt"
        },
        "VidORtrain_freq2_5k5999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq2_5k5999.txt"
        },
        "VidORtrain_freq2_6k6999":
        {
            "img_dir": "vidor-dataset/train_frames",
            "anno_path": "vidor-dataset/annotation/training",
            "img_index": "vidor-dataset/img_index/VidORtrain_freq2_6k6999.txt"
        },

    }

    @staticmethod
    def get(name, method="base"):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = deepcopy(DatasetCatalog.DATASETS[name])
            attrs["img_dir"] = os.path.join(data_dir, attrs["img_dir"])
            attrs["ann_dir"] = os.path.join(data_dir, attrs["ann_dir"])
            return dict(factory="CityScapesDataset", args=attrs)
        elif "VidOR" in name:                  # add this
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                image_set=name,
                data_dir=data_dir,
                img_dir=os.path.join(data_dir, attrs["img_dir"]),
                anno_path=os.path.join(data_dir, attrs["anno_path"]),
                img_index=os.path.join(data_dir, attrs["img_index"])  
            )
            return dict(
                factory="VidORDataset",
                args=args,
            )
        elif "COCO".upper() in name: #TODO 最好和小写coco区分的明显一些
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                image_set=name,
                data_dir=data_dir,
                img_dir=os.path.join(data_dir, attrs["img_dir"]),
                anno_path=os.path.join(data_dir, attrs["anno_path"])  
            )
            return dict(
                factory="COCOVidORDataset",
                args=args,
            )
        else:
            dataset_dict = {
                "base": "VIDDataset",
                "rdn": "VIDRDNDataset",
                "mega": "VIDMEGADataset",
                "fgfa": "VIDFGFADataset",
                "dff": "VIDDFFDataset"
            }
            if ("DET" in name) or ("VID" in name):
                data_dir = DatasetCatalog.DATA_DIR
                attrs = DatasetCatalog.DATASETS[name]
                args = dict(
                    image_set=name,
                    data_dir=data_dir,
                    img_dir=os.path.join(data_dir, attrs["img_dir"]),
                    anno_path=os.path.join(data_dir, attrs["anno_path"]),
                    img_index=os.path.join(data_dir, attrs["img_index"])
                )
                return dict(
                    factory=dataset_dict[method],
                    args=args,
                )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
