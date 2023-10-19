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

class model_LargeSimpleBaseline(nn.Module):
    def __init__(self, input_fz=133*2, output_fz=133*3):
        super(model_LargeSimpleBaseline, self).__init__()

        self.upscale = nn.Linear(input_fz, 1024)
        self.res_1 = lifter_res_block()
        self.res_2 = lifter_res_block()
        self.res_3 = lifter_res_block()
        self.pose3d = nn.Linear(1024, output_fz)

    def forward(self, p2d):

        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_1(x))
        x = nn.LeakyReLU()(self.res_2(x))
        x = nn.LeakyReLU()(self.res_3(x))
        x = self.pose3d(x)

        return x



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model_LargeSimpleBaseline().to(device)
    if os.path.exists('net.pth'):
        net.load_state_dict(torch.load('net.pth', map_location=device))
        print('Load weight successful')
    else:
        print('No pretrained net found')

    batchsize = 2
    a = torch.rand(batchsize, 133*2).to(device)
    b = net(a)
    print(b.shape)
