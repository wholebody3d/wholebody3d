# imports
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "./datasets/json/"

# load json
# task1+2_train.json is quite big. In case of memory out, load it by parts. See json_loader_part()
# id_list is to record id for each image in dataset to help finding correspondence between input and target. You need to
# write out id into your json during test in order to allow online evaluation
def json_loader(data_path, task=1, type='train'):
    id_list = []
    input_list = []
    target_list = []
    bbox_list = []
    if type == 'train':
        if (task == 1) or (task == 2)  or ('2D' in task):
            data = json.load(open(data_path+'/2Dto3D_train.json'))
            length = len(data)
            for i in range(length):
                sample_2d = torch.zeros(1, 133, 2)
                sample_3d = torch.zeros(1, 133, 3)
                for j in range(133):
                    sample_2d[0, j, 0] = data[str(i)]['keypoints_2d'][str(j)]['x']
                    sample_2d[0, j, 1] = data[str(i)]['keypoints_2d'][str(j)]['y']
                    sample_3d[0, j, 0] = data[str(i)]['keypoints_3d'][str(j)]['x']
                    sample_3d[0, j, 1] = data[str(i)]['keypoints_3d'][str(j)]['y']
                    sample_3d[0, j, 2] = data[str(i)]['keypoints_3d'][str(j)]['z']
                input_list.append(sample_2d)
                target_list.append(sample_3d)
            return input_list, target_list
        elif (task == 3) or ('RGB' in task):
            data = json.load(open(data_path+'/RGBto3D_train.json'))
            length = len(data)
            for i in range(length):
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
                input_list.append(data[str(i)]['image_path'])
                target_list.append(sample_3d)
            return input_list, target_list, bbox_list
    elif type == 'test':
        if (task == 1)  or (('2D' in task) and ('I2D' not in task)):
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

# load task1+2_train.json by parts in case the whole file is too big
def json_loader_part(data_path):
    input_list = []
    target_list = []
    for part in range(4):
        data = json.load(open(data_path+'/2Dto3D_train_part'+str(part+1)+'.json'))
        length = len(data)
        for i in range(length):
            sample_2d = torch.zeros(1, 133, 2)
            sample_3d = torch.zeros(1, 133, 3)
            for j in range(133):
                sample_2d[0, j, 0] = data[str(i)]['keypoints_2d'][str(j)]['x']
                sample_2d[0, j, 1] = data[str(i)]['keypoints_2d'][str(j)]['y']
                sample_3d[0, j, 0] = data[str(i)]['keypoints_3d'][str(j)]['x']
                sample_3d[0, j, 1] = data[str(i)]['keypoints_3d'][str(j)]['y']
                sample_3d[0, j, 2] = data[str(i)]['keypoints_3d'][str(j)]['z']
            input_list.append(sample_2d)
            target_list.append(sample_3d)
    return input_list, target_list

def get_limb(X, Y, Z=None, id1=0, id2=1):
    if Z is not None:
        return np.concatenate((np.expand_dims(X[id1], 0), np.expand_dims(X[id2], 0)), 0), \
               np.concatenate((np.expand_dims(Y[id1], 0), np.expand_dims(Y[id2], 0)), 0), \
               np.concatenate((np.expand_dims(Z[id1], 0), np.expand_dims(Z[id2], 0)), 0)
    else:
        return np.concatenate((np.expand_dims(X[id1], 0), np.expand_dims(X[id2], 0)), 0), \
               np.concatenate((np.expand_dims(Y[id1], 0), np.expand_dims(Y[id2], 0)), 0)

# draw wholebody skeleton
# conf: which joint to draw, conf=None draw all
def draw_skeleton(vec, conf=None, pointsize=None, figsize=None, plt_show=False, save_path=None, inverse_z=True,
                  fakebbox=True, background=None):
    _, keypoint, d = vec.shape
    if keypoint==133:
        X = vec
        if (d == 3) or ((d==2) and (background==None)):
            X = X - (X[:,11:12,:]+X[:, 12:13,:])/2.0
        X=X.numpy()
        list_branch_head = [(0,1),(1,3),(0,2),(2,4), (59,64), (65,70),(71, 82),
                            (71,83),(77,87),(77,88),(88,89),(89,90),(71,90)]
        for i in range(16):
            list_branch_head.append((23+i, 24+i))
        for i in range(4):
            list_branch_head.append((40+i, 41+i))
            list_branch_head.append((45+i, 46+i))
            list_branch_head.append((54+i, 55+i))
            list_branch_head.append((83+i, 84+i))
        for i in range(3):
            list_branch_head.append((50+i, 51+i))
        for i in range(5):
            list_branch_head.append((59+i, 60+i))
            list_branch_head.append((65+i, 66+i))
        for i in range(11):
            list_branch_head.append((71+i, 72+i))

        list_branch_left_arm = [(5,7),(7,9),(9,91),(91,92),(93,96),(96,100),(100,104),(104,108),(91,108)]
        for i in range(3):
            list_branch_left_arm.append((92+i,93+i))
            list_branch_left_arm.append((96+i,97+i))
            list_branch_left_arm.append((100+i,101+i))
            list_branch_left_arm.append((104+i,105+i))
            list_branch_left_arm.append((108+i,109+i))
        list_branch_right_arm = [(6,8),(8,10),(10,112),(112,113),(114,117),(117,121),(121,125),(125,129),(112,129)]
        for i in range(3):
            list_branch_right_arm.append((113+i, 114+i))
            list_branch_right_arm.append((117+i, 118+i))
            list_branch_right_arm.append((121+i, 122+i))
            list_branch_right_arm.append((125+i, 126+i))
            list_branch_right_arm.append((129+i, 130+i))
        list_branch_body = [(5,6),(6,12),(11,12),(5,11)]
        list_branch_right_foot = [(12,14),(14,16),(16,20),(16,21),(16,22)]
        list_branch_left_foot = [(11,13),(13,15),(15,17),(15,18),(15,19)]
    else:
        print('Not implemented this skeleton')
        return 0

    if d==3:
        fig = plt.figure()
        if figsize is not None:
            fig.set_size_inches(figsize,figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.elev = 10
        ax.grid(False)

        if inverse_z:
            zdata = -X[0, :, 1]
        else:
            zdata = X[0, :, 1]
        xdata = X[0, :, 0]
        ydata = X[0, :, 2]
        if conf is not None:
            xdata*=conf[0,:].numpy()
            ydata*=conf[0,:].numpy()
            zdata*=conf[0,:].numpy()
        if pointsize is None:
            ax.scatter(xdata, ydata, zdata, c='r')
        else:
            ax.scatter(xdata, ydata, zdata, s=pointsize, c='r')

        if fakebbox:
            max_range= np.array([xdata.max() - xdata.min(), ydata.max() - ydata.min(), zdata.max() - zdata.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zdata.max() + zdata.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            if background is not None:
                WidthX = Xb[7] - Xb[0]
                WidthY = Yb[7] - Yb[0]
                WidthZ = Zb[7] - Zb[0]
                arr = np.array(background.getdata()).reshape(background.size[1], background.size[0], 3).astype(float)
                arr = arr / arr.max()
                stepX, stepZ = WidthX / arr.shape[1], WidthZ / arr.shape[0]

                X1 = np.arange(0, -Xb[0]+Xb[7], stepX)
                Z1 = np.arange(Zb[7], Zb[0], -stepZ)
                X1, Z1 = np.meshgrid(X1, Z1)
                Y1 = Z1 * 0.0 + Zb[7] + 0.01
                ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, facecolors=arr, shade=False)

        for (id1, id2) in list_branch_head:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='red')
        for (id1, id2) in list_branch_body:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='orange')
        for (id1, id2) in list_branch_left_arm:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='blue')
        for (id1, id2) in list_branch_right_arm:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='violet')
        for (id1, id2) in list_branch_left_foot:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='cyan')
        for (id1, id2) in list_branch_right_foot:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='pink')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if plt_show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path)
            plt.close()

    if d==2:
        fig = plt.figure()
        if figsize is not None:
            fig.set_size_inches(figsize,figsize)
        ax = plt.axes()
        ax.axis('off')
        if background is not None:
            im = ax.imshow(background)
            ydata = X[0, :, 1]
        else:
            ydata = -X[0, :, 1]
        xdata = X[0, :, 0]
        if conf is not None:
            xdata*=conf[0,:].numpy()
            ydata*=conf[0,:].numpy()
        if pointsize is None:
            ax.scatter(xdata, ydata, c='r')
        else:
            ax.scatter(xdata, ydata, s=pointsize, c='r')

        for (id1, id2) in list_branch_head:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='red')
        for (id1, id2) in list_branch_body:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='orange')
        for (id1, id2) in list_branch_left_arm:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='blue')
        for (id1, id2) in list_branch_right_arm:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='violet')
        for (id1, id2) in list_branch_left_foot:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='cyan')
        for (id1, id2) in list_branch_right_foot:
            if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='pink')

        if fakebbox:
            max_range = np.array([xdata.max() - xdata.min(), ydata.max() - ydata.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
            for xb, yb in zip(Xb, Yb):
                ax.plot([xb], [yb], 'w')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if plt_show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path)
            plt.close()

