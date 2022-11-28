
### Notes

- The training sets contains all samples from S1, S5, S6 and S7, including 80k {image,2D,3D} triplets.
- The test set contains all samples from S8, including 20k triplets. 
- We report results for the MPJPE (Mean Per Joint Position Error) metric in mm. 
- MPJPE for the whole-body, the body (keypoint 1-23), the face (keypoint 24-91) and the hands (keypoint 92-133) when whole-body aligned with the pelvis. - Unless stated otherwise, results are pelvis aligned.
- MPJPE for the face when it is centered on the nose, i.e.aligned with keypoint 1,
- MPJPE for the hands when hands are centered on the wrist, i.e left hand aligned with keypoint 92 and right hand aligned with keypoint 113.
- Methods with<sup>*</sup> output normalized predictions.

We use the same layout from COCO-WholeBody: [Image source](https://github.com/jin-s13/COCO-WholeBody).

<img src="imgs/Fig2_anno.png" width="300" height="300">


### 1. Results for 2D &rarr; 3D task

| Method | whole-body | body | face  | nose-aligned face | hand | wrist-aligned hand |
|--------|:------------:|:------:|:-------:|:-------------------:|:------:|:--------------------:| 
CanonPose<sup>*</sup> | 186.7 | 193.7 | 188.4 | 24.6 | 180.2 | 48.9 |
SimpleBaseline<sup>*</sup> | 125.4 | 125.7 | 115.9 | 24.6 | 140.7 | 42.5 |
CanonPose w 3D supervision<sup>*</sup> | 117.7 | 117.5 | 112.0 | 17.9 | 126.9 | 38.3 |
Large SimpleBaseline<sup>*</sup> | 112.3 | 112.6 | 110.6 | **14.6** | **114.8**| **31.7** |
Jointformer | **88.3** | **84.9** | **66.5** | 17.8 | 125.3 | 43.7 |


### 2. Results for I2D &rarr; 3D task

| Method | whole-body | body | face  | nose-aligned face | hand | wrist-aligned hand |
|--------|:------------:|:------:|:-------:|:-------------------:|:------:|:--------------------:| 
CanonPose<sup>*</sup> | 285.0 | 264.4 | 319.7 | 31.9 | 240.0 | 56.2 |
SimpleBaseline<sup>*</sup> | 268.8 | 252.0 | 227.9 | 34.0 | 344.3 | 83.4 |
CanonPose + 3D supervision<sup>*</sup> | 163.6 | 155.9 | 161.3 | 22.2 | 171.4 | 47.4 |
Large SimpleBaseline<sup>*</sup> | 131.4 | 131.6 | 120.6 | **19.8** | **148.8** | **44.8** |
Jointformer | **109.2** | **103.0** | **82.4** | **19.8** | 155.9 | 53.5 |
 

### 3. Results for RGB &rarr; 3D task

| Method | whole-body | body | face  | nose-aligned face | hand | wrist-aligned hand |
|--------|:------------:|:------:|:-------:|:-------------------:|:------:|:--------------------:| 
*RGB &rarr; 2D + 2D &rarr; 3D:* | | | | | | |
SHN+SimpleBaseline$^*$ | 182.5 | 189.6 | 138.7 | 32.5 | 249.4 | 64.3 |
CPN+Jointformer | **132.6** | **142.8** | **91.9** | **20.7** | **192.7** | **56.9** |
*RGB &rarr; 3D* | | | | | | |
Resnet50 | 166.7 | 151.6 | 123.6 | 26.3 | 244.9 | 63.1 |
