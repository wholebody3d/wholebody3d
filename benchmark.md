
### Notes

- The training sets contains all samples from S1, S5, S6 and S7, including 80k {image,2D,3D} triplets.
- The test set contains all samples from S8, including 20k triplets. 
- We do not provide a validation set. We encourage researchers to report 5-fold cross-validation average and standard deviation.

For each task, we report the following MPJPE (Mean Per Joint Position Error) metrics:
- MPJPE for the whole-body, the body (keypoint 1-23), the face (keypoint 24-91) and the hands (keypoint 92-133) when whole-body aligned with the pelvis.
- MPJPE for the face when it is centered on the nose, i.e.aligned with keypoint 1,
- MPJPE for the hands when hands are centered on the wrist, i.e left hand aligned with keypoint 92 and right hand aligned with keypoint 113.


We use the same layout from COCO-WholeBody: [Image source](https://github.com/jin-s13/COCO-WholeBody).

<img src="imgs/Fig2_anno.png" width="300" height="300">


## Results for the 3D whole-body lifting from complete 2D whole-body keypoints (2D→3D)

| Method | whole-body | body | face  | nose-aligned face | hand | wrist-aligned hand |
|--------|------------|------|-------|-------------------|------|--------------------| 
CanonPose<sub>*</sub> | 186.7 | 193.7 | 188.4 | 24.6 | 180.2 | 48.9 |
SimpleBaseline | 125.4 | 125.7 | 115.9 | 24.6 | 140.7 | 42.5 |
CanonPose w 3D supervision | 117.7 | 117.5 | 112.0 | 17.9 | 126.9 | 38.3 |
Large SimpleBaseline | 112.3 | 112.6 | 110.6 | **14.6** | **114.8**| **31.7** |
Jointformer | **88.3** | **84.9** | **66.5** | 17.8 | 125.3 | 43.7 |


- Results are shown for the MPJPE metric in mm. 
- Methods with ∗ output normalized predictions.
- Unless stated otherwise, results are pelvis aligned.
