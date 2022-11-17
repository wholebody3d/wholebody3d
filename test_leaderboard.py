# imports
import os
import sys
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cross_validation =
# 0: train/test split, 1: 0-15999 of train sample as val, ..., 5: 64000-79999 of train sample as val
def data_loader(data_path, task=1, type='train', cross_validation=0):
    id_list = []
    input_list = []
    target_list = []
    bbox_list = []
    if type == 'train':
        if (task == 1) or (task == 2) or ('2D' in task):
            data = json.load(open(data_path+'/2Dto3D_train.json'))
            length = len(data)
            for i in range(length):
                if (cross_validation == 0) or (((length//5) * (cross_validation-1)<=i) and ((length//5) * (cross_validation)>i)):
                    sample_2d = torch.zeros(1, 133, 2)
                    sample_3d = torch.zeros(1, 133, 3)
                    for j in range(133):
                        sample_2d[0, j, 0] = data[str(i)]['keypoints_2d'][str(j)]['x']
                        sample_2d[0, j, 1] = data[str(i)]['keypoints_2d'][str(j)]['y']
                        sample_3d[0, j, 0] = data[str(i)]['keypoints_3d'][str(j)]['x']
                        sample_3d[0, j, 1] = data[str(i)]['keypoints_3d'][str(j)]['y']
                        sample_3d[0, j, 2] = data[str(i)]['keypoints_3d'][str(j)]['z']
                    id_list.append(i)
                    input_list.append(sample_2d)
                    target_list.append(sample_3d)
            return id_list, input_list, target_list
        elif (task == 3) or ('RGB' in task):
            data = json.load(open(data_path+'/RGBto3D_train.json'))
            length = len(data)
            for i in range(length):
                if (cross_validation == 0) or (((length // 5) * (cross_validation - 1) <= i) and ((length // 5) * (cross_validation) > i)):
                    sample_3d = torch.zeros(1, 133, 3)
                    bbox = torch.zeros(1,4)
                    bbox[0, 0] = int(data[str(i)]['bbox']['x_min'])
                    bbox[0, 1] = int(data[str(i)]['bbox']['y_min'])
                    bbox[0, 2] = int(data[str(i)]['bbox']['x_max'])
                    bbox[0, 3] = int(data[str(i)]['bbox']['y_max'])
                    bbox_list.append(bbox)
                    for j in range(133):
                        sample_3d[0, j, 0] = data[str(i)]['keypoints_3d'][str(j)]['x']
                        sample_3d[0, j, 1] = data[str(i)]['keypoints_3d'][str(j)]['y']
                        sample_3d[0, j, 2] = data[str(i)]['keypoints_3d'][str(j)]['z']
                    id_list.append(i)
                    input_list.append(data[str(i)]['image_path'])
                    target_list.append(sample_3d)
            return id_list, input_list, target_list, bbox_list
    elif type == 'test':
        if (task == 1) or (('2D' in task) and ('I2D' not in task)):
            data = json.load(open(data_path+'/2Dto3D_test_2d.json'))
            length = len(data)
            for i in range(length):
                sample_2d = torch.zeros(1, 133, 2)
                for j in range(133):
                    sample_2d[0, j, 0] = data[str((i//4)*8+(i%4))]['keypoints_2d'][str(j)]['x']
                    sample_2d[0, j, 1] = data[str((i//4)*8+(i%4))]['keypoints_2d'][str(j)]['y']
                id_list.append((i//4)*8+(i%4))
                input_list.append(sample_2d)
            return id_list, input_list
        elif (task == 2) or ('I2D' in task):
            data = json.load(open(data_path+'/I2Dto3D_test_2d.json'))
            length = len(data)
            for i in range(length):
                sample_2d = torch.zeros(1, 133, 2)
                for j in range(133):
                    sample_2d[0, j, 0] = data[str((i//4)*8+(i%4)+4)]['keypoints_2d'][str(j)]['x']
                    sample_2d[0, j, 1] = data[str((i//4)*8+(i%4)+4)]['keypoints_2d'][str(j)]['y']
                id_list.append((i//4)*8+(i%4)+4)
                input_list.append(sample_2d)
            return id_list, input_list
        elif (task == 3) or ('RGB' in task):
            data = json.load(open(data_path+'/RGBto3D_test_img.json'))
            length = len(data)
            for i in range(length):
                id_list.append(i)
                input_list.append(data[str(i)]['image_path'])
                bbox = torch.zeros(1, 4)
                bbox[0, 0] = int(data[str(i)]['bbox']['x_min'])
                bbox[0, 1] = int(data[str(i)]['bbox']['y_min'])
                bbox[0, 2] = int(data[str(i)]['bbox']['x_max'])
                bbox[0, 3] = int(data[str(i)]['bbox']['y_max'])
                bbox_list.append(bbox)
            return id_list, input_list, bbox_list
    elif type == 'admin':
        if (task == 1) or (('2D' in task) and ('I2D' not in task)):
            data = json.load(open(data_path+'/2Dto3D_test_3d.json'))
            length = len(data)
            for i in range(length):
                sample_2d = torch.zeros(1, 133, 2)
                sample_3d = torch.zeros(1, 133, 3)
                for j in range(133):
                    sample_2d[0, j, 0] = data[str((i//4)*8+(i%4))]['keypoints_2d'][str(j)]['x']
                    sample_2d[0, j, 1] = data[str((i//4)*8+(i%4))]['keypoints_2d'][str(j)]['y']
                    sample_3d[0, j, 0] = data[str((i//4)*8+(i%4))]['keypoints_3d'][str(j)]['x']
                    sample_3d[0, j, 1] = data[str((i//4)*8+(i%4))]['keypoints_3d'][str(j)]['y']
                    sample_3d[0, j, 2] = data[str((i//4)*8+(i%4))]['keypoints_3d'][str(j)]['z']
                id_list.append((i//4)*8+(i%4))
                input_list.append(sample_2d)
                target_list.append(sample_3d)
            return id_list, input_list, target_list
        elif (task == 2) or ('I2D' in task):
            data = json.load(open(data_path+'/I2Dto3D_test_3d.json'))
            length = len(data)
            for i in range(length):
                sample_2d = torch.zeros(1, 133, 2)
                sample_3d = torch.zeros(1, 133, 3)
                for j in range(133):
                    sample_2d[0, j, 0] = data[str((i//4)*8+(i%4)+4)]['keypoints_2d'][str(j)]['x']
                    sample_2d[0, j, 1] = data[str((i//4)*8+(i%4)+4)]['keypoints_2d'][str(j)]['y']
                    sample_3d[0, j, 0] = data[str((i//4)*8+(i%4)+4)]['keypoints_3d'][str(j)]['x']
                    sample_3d[0, j, 1] = data[str((i//4)*8+(i%4)+4)]['keypoints_3d'][str(j)]['y']
                    sample_3d[0, j, 2] = data[str((i//4)*8+(i%4)+4)]['keypoints_3d'][str(j)]['z']
                id_list.append((i//4)*8+(i%4)+4)
                input_list.append(sample_2d)
                target_list.append(sample_3d)
            return id_list, input_list, target_list
        elif (task == 3) or ('RGB' in task):
            data = json.load(open(data_path+'/RGBto3D_test_3d.json'))
            length = len(data)
            for i in range(length):
                sample_3d = torch.zeros(1, 133, 3)
                bbox = torch.zeros(1, 4)
                bbox[0, 0] = int(data[str(i)]['bbox']['x_min'])
                bbox[0, 1] = int(data[str(i)]['bbox']['y_min'])
                bbox[0, 2] = int(data[str(i)]['bbox']['x_max'])
                bbox[0, 3] = int(data[str(i)]['bbox']['y_max'])
                bbox_list.append(bbox)
                for j in range(133):
                    sample_3d[0, j, 0] = data[str(i)]['keypoints_3d'][str(j)]['x']
                    sample_3d[0, j, 1] = data[str(i)]['keypoints_3d'][str(j)]['y']
                    sample_3d[0, j, 2] = data[str(i)]['keypoints_3d'][str(j)]['z']
                id_list.append(i)
                input_list.append(data[str(i)]['image_path'])
                target_list.append(sample_3d)
            return id_list, input_list, target_list, bbox_list

def test_score(data_path):
    print(data_path)
    task = 1
    for i in range(3):
        if 'task' + str(i + 1) in data_path:
            task = i + 1
    if 'RGB' in data_path:
        task = 3
    elif 'I2D' in data_path:
        task = 2
    elif '2D' in data_path:
        task = 1

    cross_validation = 0
    for i in range(6):
        if 'cv' + str(i) in data_path:
            cross_validation = i

    predict_data = json.load(open(data_path))
    gt_data_path = './datasets/json/'
    if cross_validation == 0:
        if task <3:
            id_list, _, target_list = data_loader(gt_data_path, task=task, type='admin')
        else:
            id_list, _, target_list, _ = data_loader(gt_data_path, task=task, type='admin')
    else:
        if task <3:
            id_list, _, target_list = data_loader(gt_data_path, task=task, type='train', cross_validation=cross_validation)
        else:
            id_list, _, target_list, _ = data_loader(gt_data_path, task=task, type='train', cross_validation=cross_validation)

    predict_list = []
    for i in range(len(id_list)):
        try:
            sample_3d = torch.zeros(1, 133, 3)
            for j in range(133):
                sample_3d[0, j, 0] = predict_data[str(id_list[i])]['keypoints_3d'][str(j)]['x']
                sample_3d[0, j, 1] = predict_data[str(id_list[i])]['keypoints_3d'][str(j)]['y']
                sample_3d[0, j, 2] = predict_data[str(id_list[i])]['keypoints_3d'][str(j)]['z']
            predict_list.append(sample_3d)
        except ValueError:
            print("Could not find prediction for id", id_list[i])
            return 1

    predict_list = torch.cat(predict_list, dim=0)
    target_list = torch.cat(target_list, dim=0)

    count = [0,0,0,0,0,0]
    diff = predict_list - target_list
    diff = diff - (diff[:, 11:12, :] + diff[:, 12:13, :]) / 2  # pelvis align
    diff1 = (diff - diff[:, 0:1, :])[:, 23:91, :]  # nose align face
    diff21 = (diff - diff[:, 91:92, :])[:, 91:112, :]  # wrist aligned left hand
    diff22 = (diff - diff[:, 112:113, :])[:, 112:, :]  # wrist aligned right hand

    diff = torch.sqrt(torch.sum(torch.square(diff), dim=-1))

    count[0] = torch.mean(diff).item()
    count[1] = torch.mean(diff[:, :23]).item()
    count[2] = torch.mean(diff[:, 23:91]).item()
    count[3] = torch.mean(diff[:, 91:]).item()
    count[4] = torch.mean(torch.sqrt(torch.sum(torch.square(diff1), dim=-1))).item()
    count[5] = torch.mean(torch.sqrt(torch.sum(torch.square(diff21), dim=-1))).item() \
               + torch.mean(torch.sqrt(torch.sum(torch.square(diff22), dim=-1))).item()

    print("Pelvis aligned MPJPE is " + str(count[0]) + ' mm')
    print("Pelvis aligned MPJPE on body is " + str(count[1]) + ' mm')
    print("Pelvis aligned MPJPE on face is " + str(count[2]) + ' mm')
    print("Nose aligned MPJPE on face is " + str(count[4]) + ' mm')
    print("Pelvis aligned MPJPE on hands is " + str(count[3]) + ' mm')
    print("Wrist aligned MPJPE on hands is " + str(count[5]/2) + ' mm')


if __name__ == "__main__":
    args = sys.argv[1:]
    test_score(args[-1])