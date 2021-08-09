official format of the VidVRD/VidOR annotations (with some additional notes)


```json5
{
    "video_id": "ILSVRC2015_train_00010001",        # Video ID from the original ImageNet ILSVRC2016 video dataset
    "frame_count": 219,
    "fps": 30, 
    "width": 1920, 
    "height": 1080, 
    "subject/objects": [                            # List of subject/objects, 
        {                                           # tids do not necessarily cover 0 ~ len(traj_categories)-1
            "tid": 0,                               # Trajectory ID of a subject/object
            "category": "bicycle"
        }, 
        ...
    ], 
     "trajectories": [                              # List of frames   
        [                                           # List of bounding boxes in each frame, this list can be [] (empty)
            {
                "tid": 0,                       
                "bbox": {
                    "xmin": 672,                    # left
                    "ymin": 560,                    # top
                    "xmax": 781,                    # right
                    "ymax": 693                     # bottom
                }, 
                "generated": 0,                     # 0 - the bounding box is manually labeled
                                                    # 1 - the bounding box is automatically generated
            },  
            ...
        ],
        ...
    ]
    "relation_instances": [                         # List of annotated visual relation instances
        {
            "subject_tid": 0,                       # Corresponding trajectory ID of the subject
            "object_tid": 1,                        # Corresponding trajectory ID of the object
            "predicate": "move_right", 
            "begin_fid": 0,                         # Frame index where this relation begins (inclusive)
            "end_fid": 210                          # Frame index where this relation ends (exclusive)
        }, 
        ...
    ]
}
```