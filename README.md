# Human3.6m 3D-wholebody extension dataset

This is the official repository for the paper ["TBD Paper name"](TBD).

## What do we provide
This is an extension of [Human3.6m dataset](http://vision.imar.ro/human3.6m/) which contains 100k image-2D-3D wholebody 
annotations of 133 joints each. The skeleton layout is the same as 
[COCO-Wholebody dataset](https://github.com/jin-s13/COCO-WholeBody).

An example of the annotations:

<img src="imgs/1.jpg" width="800" height="400">

<img src="imgs/2.gif" width="600" height="600">

## How to use

### Download
Images need to be downloaded at the official cite of
[Human3.6m dataset](http://vision.imar.ro/human3.6m/).
The annotations are in this repository in "./datasets/json/"

### Annotation format
Every json is in the following structure, but not every json contains all these values. See Task section.
```
XXX.json --- sample id --- 'image_path'
                        |
                        -- 'bbox' --- 'x_min'
                        |          |- 'y_min'
                        |          |- 'x_max'
                        |          |- 'y_max'
                        |
                        |- 'keypont_2d' --- joint id --- 'x'
                        |                             |- 'y'
                        |
                        |- 'keypont_3d' --- joint id --- 'x'
                                                      |- 'y'
                                                      |- 'z'
                        
                        
```
We provide function in './utils/utils.py' to load json files

### Task

We propose 3 different tasks to achieve with this extension:

#### Task 1: 2D complete wholebody to 3D complete wholebody lifting

 - Use task1+2_train.json for training/validation. It contains 80k keypoint_2d and keypoint_3d

 - Use task1_test_2d.json for test on leaderboard. It contains 10k keypoint_2d

#### Task 2: 2D incomplete wholebody to 3D complete wholebody lifting

 - Use task1+2_train.json for training/validation. It contains 80k keypoint_2d and keypoint_3d

 - Please apply masking on yourself during the training. The official masking strategy is: 40% chance that each joint has 25% 
chance being masked; otherwise 20% chance face masking; 
20% chance left hand masking; and 20% chance right hand masking.

 - Use task2_test_2d.json for test on leaderboard. It contains 10k keypoint_2d
 - To avoid cheating, this test set is not the same as Task 1, as well as already having mask on keypoint_2d

#### Task 3 Image to 3D complete wholebody prediction

 - Use task3_train.json for training/validation. It contains 80k image_path, bounding box and keypoint_3d
 - It use same sample id as task1+2_train.json, so you can also find keypoint_2D if needed

 - Use task3_test_img.json for test on leaderboard. It contains 20k image_path and bounding box. (Test sample of task1 + 
task2.)
 - To avoid cheating, the sample id are not aligned with previous 2 tasks with some kind of random permutation)

### Evaluation

Please save your 3D wholebody predictions on test set into 'taskX_pred.json' using same data format as given one and
submit [here]()

We provide a function to visualize 3D wholebody, as well as the same evaluation function for the leaderboard in 
'.utils/utils.py'

### Terms of Use

1. This dataset is **ONLY** for research and non-commercial use. 
   The annotations of this dataset belong to [TBD](), and are licensed under a [TBD]().

2. For commercial usage, please contact [Anonymous for now]()

3. We do not own the copyright of the images. Use of the images must abide by the 
   [Human3.6m License agreement](http://vision.imar.ro/human3.6m/eula.php).

## Benchmark

| Method | Task | Whole | Body | Raw Face  | Nose-aligned Face | Raw Hands | Wrist-aligned Hands |
|--------|------|-------|------|-----------|-------------------|-----------|---------------------|

## Citation

If you use this dataset in your project, please cite this paper.

```
@inproceedings{XXX,
  title={},
  author={Anonymous for now},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition (CVPR)},    
  year={2022}
}
```

## Reference

```
```