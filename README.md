
# VidVRD-tracklets
This repository contains codes for Video Visual Relation Detection (VidVRD) tracklets generation based on [MEGA](https://github.com/Scalsol/mega.pytorch) and [deepSORT](https://github.com/nwojke/deep_sort). These tracklets are also suitable for ACM MM Visual Relation Understanding (VRU) Grand Challenge. 

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
│   ├── ...
│   └── ...
```
Please refer to `deepSORT/tracking_results/readme.md` for more details


## Generate object tracklets by yourself

The object tracklets generation pipeline mainly consists of two parts: ``MEGA`` (for video object detection), and ``deepSORT`` (for video object tracking). 