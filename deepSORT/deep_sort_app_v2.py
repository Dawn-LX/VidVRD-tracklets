# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# from IPython import embed


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
            # detections.shape == (438,2058)
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    # image_dir = os.path.join(sequence_dir, "img1")
    # print(image_dir,"--------image_dir--------")
    image_dir = sequence_dir  # sequence_dir='videovrd-dataset/frames/ILSVRC2015_train_00010001'
    image_filenames = {
        int(f.split(".")[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}   # frame_id to image_filename
    
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:    # detection_file = videovrd-dataset/det_res_dir/ILSVRC2015_train_00010001/ILSVRC2015_train_00010001.npy
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    # print("info_filename",info_filename)
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0   
    # detections.shape = (438,2058)  # feature_dim=2048, RoI feature from FasterRCNN
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray,  shape = (438, 2058) 438 代表这个video中所有frame 中的 总bbox数量
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)  # 每个bbox所在frame 的index
    mask = frame_indices == frame_idx  # 找到当前frame 的 bbox

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, label, feature = row[2:6], row[6], row[7], row[10:]   # PKU_cvpr2020, 在原版deep_sort的基础上进行了一些修改
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, label, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display,max_miss,seq_info=None):
    """Run multi-target tracker on a particular sequence.

    Parameters
        ----------
        sequence_dir : str  sequence_dir='videovrd-dataset/frames/ILSVRC2015_train_00010001'
            Path to the MOTChallenge sequence directory.
        detection_file : str
            Path to the detections file.
            detections = np.load(detection_file), detections.shape == (438,2058)
            # # shape == 438,2058  # 每个 bbox 对应的feature vector 是2058维的, 438 是所有的frame中bbox的总数
            # 2058 这个维度中，其中2048维是RoI feature，另外10个维度中包含了frame_id, label, bbox_coordinate, score
        output_file : str
            Path to the tracking output file. This file will contain the tracking
            results on completion.
        min_confidence : float
            Detection confidence threshold. Disregard all detections that have  #Disregard: 拒绝，排除
            a confidence lower than this value.
        nms_max_overlap: float
            Maximum detection overlap (non-maxima suppression threshold).
        min_detection_height : int
            Detection height threshold. Disregard all detections that have
            a height lower than this value.
        max_cosine_distance : float
            Gating threshold for cosine distance metric (object appearance).
        nn_budget : Optional[int]
            Maximum size of the appearance descriptor gallery. If None, no budget
            is enforced.
        display : bool
            If True, show visualization of intermediate tracking results.

    """
    if seq_info == None:
        assert (sequence_dir!=None) and (detection_file!=None)
        seq_info = gather_sequence_info(sequence_dir, detection_file)
    
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric,max_age=max_miss) # 默认允许miss30帧
    results = []
    # 以每个 bbox 的RoI feature 作为距离做匹配，
    # 他的deep_sort 中的deep 就是指用了CNN计算得到的feature， PKU_cvpr2020 就是把原来deep_sort中的CNN feature 换成了 RoI feature

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        # detections 是一个list，每个元素是一个class Detection 的对象，每个对象存放一个 bbox，以及相应的confidence, label, feature
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()  # 将 tracker.matches 置为空list
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        track2det = dict()
        # print(tracker.matches,"tracker.matches-----------")

        for ele in tracker.matches:
            track2det[ele[0]] = ele[1]
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            track_res = [frame_idx, track.track_id, bbox[0], bbox[1], bbox[2],
                    bbox[3]]
            if i not in track2det.keys():
                results.append(track_res)
                continue
            track_res += [detections[track2det[i]].confidence]
            track_res += [detections[track2det[i]].label]
            track_res += list(detections[track2det[i]].tlwh)
            track_res += list(detections[track2det[i]].feature)
            results.append(track_res)
            # results.append([
                # frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3],
                # detection])
        
        # frame_callback这个回调函数是逐帧调用的，
        # track_id 是轨迹的id， 比如第一帧有四个object，那首先track_id 就是 1,2,3,4; 然后第二帧如
        # 果还是4个box，那这4个box和第一帧的四个算相似度，如果是很相似的，那就第二帧的track_id还是1,2,3,4
        # 再然后如果第三帧还是4个box，其中3个是能并入前面的轨迹的，那track_id 1,2,3不变， 然后第四个box并入不了前
        # 面的轨迹，那这时候这个box的track_id 就编为5， 也就是说，4号轨迹在这一帧就中断了

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    # f = open(output_file, 'w')
    # for row in results:
        # f.writelines("%d,%d" % (row[0], row[1]))
        # for ele in row[2:]:
            # f.writelines(",%.64f" % ele)
        # f.writelines("\n")
    # f.close()
    res = np.array(results)
    np.save(output_file, res)
    #for row in results:
    #    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
    #        row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=False, type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # print("in deep_sort_app")
    # print(args)

    # detection_file='videovrd-dataset/det_res_dir/ILSVRC2015_train_00010001/ILSVRC2015_train_00010001.npy', 
    # display=False, 
    # max_cosine_distance=0.2, 
    # min_confidence=0.3, 
    # min_detection_height=0, 
    # nms_max_overlap=1.0, 
    # nn_budget=None, 
    # output_file='videovrd-dataset/videovrd_tracking_res/ILSVRC2015_train_00010001.npy', 
    # sequence_dir='videovrd-dataset/frames/ILSVRC2015_train_00010001'

    ### default args:
    # nms_max_overlap = 1.0
    # min_detection_height = 0
    # max_cosine_distance = 0.2
    # nn_budget = None
    # display = False

    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, display=False)
