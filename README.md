
# VidVRD-tracklets
We won the 1st place of Video Relation Understanding (VRU) Grand Challenge in ACM Multimedia 2021. The corresponding technical report: [here](https://dl.acm.org/doi/10.1145/3474085.3479231) or [arXiv version](https://arxiv.org/abs/2108.08669).

This repository contains codes for Video Visual Relation Detection (VidVRD) tracklets generation based on [MEGA](https://github.com/Scalsol/mega.pytorch) and [deepSORT](https://github.com/nwojke/deep_sort). These tracklets are also suitable for [ACM MM Visual Relation Understanding (VRU) Grand Challenge](https://videorelation.nextcenter.org/) (which is base on the [VidOR dataset](https://xdshang.github.io/docs/vidor.html)).

If you are only interested in the generated tracklets, ​you can ignore these codes and download them directly from [here](https://drive.google.com/drive/folders/1wWkzHlhYcZPQR4fUMTTJEn2SVVnhGFch?usp=sharing)


## Download generated tracklets directly
We release the object tracklets for [VidOR](https://xdshang.github.io/docs/vidor.html) train/validation/test set. You can download the tracklets [here](https://drive.google.com/drive/folders/1wWkzHlhYcZPQR4fUMTTJEn2SVVnhGFch?usp=sharing), and put them in the following folder as 

```
├── deepSORT
│   ├── ...
│   ├── tracking_results
│   │   ├── VidORtrain_freq1_m60s0.3_part01
│   │   ├── ...
│   │   ├── VidORtrain_freq1_m60s0.3_part14
│   │   ├── VidORval_freq1_m60s0.3
│   │   ├── VidORtest_freq1_m60s0.3
│   │   ├── readme.md
│   │   └── format_demo.py
│   └── ...
├── MEGA
│   ├── ... 
│   └── ...
```
Please refer to `deepSORT/tracking_results/readme.md` for more details

## Evaluate the tracklets mAP
Run `python deepSORT/eval_traj_mAP.py` to evaluate the tracklets mAP. (you might need to change some args in deepSORT/eval_traj_mAP.py)

## Generate object tracklets by yourself

The object tracklets generation pipeline mainly consists of two parts: ``MEGA`` (for video object detection), and ``deepSORT`` (for video object tracking). 

### Quick Start

1. Install MEGA as the official instructions `MEGA/INSTALL.md`  (Note that the folder path may be different when installing). 
    
    - If you have any trouble when installing MEGA, you can try to clone the [official MEGA repository](https://github.com/Scalsol/mega.pytorch) and install it, and then replace the official `mega.pytorch/mega_core` with our modified `MEGA/mega_core`. Refer to `MEGA/modification_details.md` for the details of our modifications.


2. Download the [VidOR dataset](https://xdshang.github.io/docs/vidor.html) and the pre-trained [weight](https://drive.google.com/file/d/1nypbyRLpiQkxr7jvnnM4LEx2ZJuzrjws/view?usp=sharing) of MEGA. Put them in the following folder as 

```
├── deepSORT/
│   ├── ...
├── MEGA/
│   ├── ... 
│   ├── datasets/
│   │   ├── COCOdataset/        # used for MEGA training
│   │   ├── COCOinVidOR/        # used for MEGA training
│   │   ├── vidor-dataset/
│   │   │   ├── annotation/
│   │   │   │   ├── training/
│   │   │   │   └── validation/
│   │   │   ├── img_index/ 
│   │   │   │   ├── VidORval_freq1_0024.txt
│   │   │   │   ├── ...
│   │   │   ├── val_frames/
│   │   │   │   ├── 0001_2793806282/
│   │   │   │   │   ├── 000000.JPEG
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── val_videos/
│   │   │   │   ├── 0001/
│   │   │   │   │   ├── 2793806282.mp4
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── train_frames/
│   │   │   ├── train_videos/
│   │   │   ├── test_frames/
│   │   │   ├── test_videos/
│   │   │   └── video2img_vidor.py
│   │   └── construct_img_idx.py
│   ├── training_dir/
│   │   ├── COCO34ORfreq32_4gpu/
│   │   │   ├── inference/
│   │   │   │   ├── VidORval_freq1_0024/
│   │   │   │   │   ├── predictions.pth
│   │   │   │   │   └── result.txt
│   │   │   │   ├── ...
│   │   │   └── model_0180000.pth
│   │   ├── ...
```

3. Run `python MEGA/datasets/vidor-dataset/video2img_vidor.py` (note that you may need to change some args) to extract frames from videos (This causes a lot of data redundancy, but we have to do this, because MEGA takes image data as input). 

4. Run `python MEGA/datasets/construct_img_idx.py` (note that you may need to change some args) to generate the img_index used in MEGA inference.
    - The generated `.txt` files will be saved in `MEGA/datasets/vidor-dataset/img_index/`. You can use `VidORval_freq1_0024.txt` as a demo for the following commands.

5. Run the following command to detect frame-level object proposals with bbox features (RoI pooled features).

    ```
    CUDA_VISIBLE_DEVICES=0   python  \
        MEGA/tools/test_net.py \
        --config-file MEGA/configs/MEGA/inference/VidORval_freq1_0024.yaml \
        MODEL.WEIGHT MEGA/training_dir/COCO34ORfreq32_4gpu/model_0180000.pth \
        OUTPUT_DIR MEGA/training_dir/COCO34ORfreq32_4gpu/inference
    ```
    - The above command will generate a `predictions.pth` file for this `VidORval_freq1_0024` demo. We also release this `predictions.pth` [here](https://drive.google.com/drive/folders/1l0pcOGycs6fnmMQu2RLK8Vu_iD9TG2zO?usp=sharing). 

    - the config files for VidOR train set are in `MEGA/configs/MEGA/partxx`

    - The `predictions.pth` contains frame-level box positions and features (RoI features) for each object. For RoI features, they can be accessed through `roifeats = boxlist.get_field("roi_feats")`, if you are familiar with MEGA or [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

6. Run `python MEGA/mega_boxfeatures/cvt_proposal_result.py` (note that you may need to change some args) to convert `predictions.pth` to a `.pkl` file for the following deepSORT stage.
    - We also provide `VidORval_freq1_0024.pkl` [here](https://drive.google.com/file/d/1r9WZG6pXXk8IT7E1E8oqJ4zBl38uYiqq/view?usp=sharing)

6. Run `python deepSORT/deepSORT_tracking_v2.py` (note that you may need to change some args) to perform deepSORT tracking. The results will be saved in `deepSORT/tracking_results/`

## Train MEGA for VidOR by yourself

1. Download MS-COCO and put them as shown in above.

2. Run `python MEGA/tools/extract_coco.py` to extract annotations for COCO in VidOR, which results in `COCO_train_34classes.pkl` and `COCO_valmini_34classes.pkl`

3. train MEGA by the following commands:

```
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        tools/train_net.py \
        --master_port=$((RANDOM + 10000)) \
        --config-file MEGA/configs/MEGA/vidor_R_101_C4_MEGA_1x_4gpu.yaml \
        OUTPUT_DIR MEGA/training_dir/COCO34ORfreq32_4gpu
```

More detailed training instructions will be updated soon...