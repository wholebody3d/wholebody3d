
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
CanonPose [[2]](#2)<sup>*</sup> | 186.7 | 193.7 | 188.4 | 24.6 | 180.2 | 48.9 |
SimpleBaseline [[1]](#1)<sup>*</sup> | 125.4 | 125.7 | 115.9 | 24.6 | 140.7 | 42.5 |
CanonPose [[2]](#2) with 3D supervision<sup>*</sup> | 117.7 | 117.5 | 112.0 | 17.9 | 126.9 | 38.3 |
Large SimpleBaseline [[1]](#1)<sup>*</sup> | 112.3 | 112.6 | 110.6 | **14.6** | **114.8**| **31.7** |
Jointformer [[3]](#3) | **88.3** | **84.9** | **66.5** | 17.8 | 125.3 | 43.7 |


### 2. Results for I2D &rarr; 3D task

| Method | whole-body | body | face  | nose-aligned face | hand | wrist-aligned hand |
|--------|:------------:|:------:|:-------:|:-------------------:|:------:|:--------------------:| 
CanonPose [[2]](#2)<sup>*</sup> | 285.0 | 264.4 | 319.7 | 31.9 | 240.0 | 56.2 |
SimpleBaseline [[1]](#1)<sup>*</sup> | 268.8 | 252.0 | 227.9 | 34.0 | 344.3 | 83.4 |
CanonPose [[2]](#2) with 3D supervision<sup>*</sup> | 163.6 | 155.9 | 161.3 | 22.2 | 171.4 | 47.4 |
Large SimpleBaseline [[1]](#1)<sup>*</sup> | 131.4 | 131.6 | 120.6 | **19.8** | **148.8** | **44.8** |
Jointformer [[3]](#3) | **109.2** | **103.0** | **82.4** | **19.8** | 155.9 | 53.5 |
 

### 3. Results for RGB &rarr; 3D task

| Method | whole-body | body | face  | nose-aligned face | hand | wrist-aligned hand |
|--------|:------------:|:------:|:-------:|:-------------------:|:------:|:--------------------:| 
SHN [[4]](#4) + SimpleBaseline [[1]](#1)<sup>*</sup> | 182.5 | 189.6 | 138.7 | 32.5 | 249.4 | 64.3 |
CPN [[5]](#5) + Jointformer [[3]](#3) | **132.6** | **142.8** | **91.9** | **20.7** | **192.7** | **56.9** |
Resnet50 [[6]](#6) | 166.7 | 151.6 | 123.6 | 26.3 | 244.9 | 63.1 |


#### References
<a id="1">[1]</a> Julieta Martinez, Rayat Hossain, Javier Romero, and James J. Little. A simple yet effective baseline for 3d human pose estimation. ICCV, 2017. \
<a id="2">[2]</a> Bastian Wandt, Marco Rudolph, Petrissa Zell, Helge Rhodin, and Bodo Rosenhahn. Canonpose: Self-supervised monocular 3D human pose estimation in the wild. CVPR, 2021.  \
<a id="3">[3]</a> Sebastian Lutz, Richard Blythman, Koustav Ghosal, Matthew Moynihan, Ciaran Simms, and Aljosa Smolic. Jointformer: Single-frame lifting transformer with error prediction and refinement for 3d human pose estimation. ArXiv, 2022. \
<a id="4">[4]</a> Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. ECCV, 2016.\
<a id="5">[5]</a> Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network for multi-person pose estimation. CVPR, 2017.\
<a id="6">[6]</a> Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CVPR, 2015.
