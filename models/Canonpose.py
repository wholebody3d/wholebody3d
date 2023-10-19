import os
import torch
import torch.nn as nn

class lifter_res_block(nn.Module):
    def __init__(self, hidden=1024):
        super(lifter_res_block, self).__init__()
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp

        return x

class model_lifter_canonpose(nn.Module):
    def __init__(self, input_fz=133*3, output_fz=133*3):
        super(model_lifter_canonpose, self).__init__()

        self.upscale = nn.Linear(input_fz, 1024)
        self.res_common = lifter_res_block()
        self.res_pose1 = lifter_res_block()
        self.res_pose2 = lifter_res_block()
        self.res_cam1 = lifter_res_block()
        self.res_cam2 = lifter_res_block()
        self.pose3d = nn.Linear(1024, output_fz)
        self.enc_rot = nn.Linear(1024, 3)

    def forward(self, p2d, conf):

        x = torch.cat((p2d, conf), axis=1)

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        x_pose = self.pose3d(xp)

        # camera path
        xc = nn.LeakyReLU()(self.res_cam1(x))
        xc = nn.LeakyReLU()(self.res_cam2(xc))
        xc = self.enc_rot(xc)

        return x_pose, xc




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model_lifter_canonpose().to(device)
    if os.path.exists('net.pth'):
        net.load_state_dict(torch.load('net.pth', map_location=device))
        print('Load weight successful')
    else:
        print('No pretrained net found')

    batchsize = 2
    a = torch.rand(batchsize, 133*2).to(device)
    conf = torch.rand(batchsize, 133*1)
    b, c = net(a, conf)
    print(b.shape)
    print(c.shape)
