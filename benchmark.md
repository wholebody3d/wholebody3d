
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
