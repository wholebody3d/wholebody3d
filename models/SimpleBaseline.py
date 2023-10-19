import os
import torch
import torch.nn as nn

class model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation(nn.Module):
    def __init__(self):
        super(model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation, self).__init__()
        self.upscale = nn.Linear(133*2, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.outputlayer = nn.Linear(1024, 133*3)

    def forward(self, x):
        x = self.upscale(x)
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn1(self.fc1(x))))
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn2(self.fc2(x1))))
        x = x + x1
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn3(self.fc3(x))))
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn4(self.fc4(x1))))
        x = x + x1
        x = self.outputlayer(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation().to(device)
    if os.path.exists('net.pth'):
        net.load_state_dict(torch.load('net.pth', map_location=device))
        print('Load weight successful')
    else:
        print('No pretrained net found')

    batchsize = 2
    a = torch.rand(batchsize, 133*2).to(device)
    b = net(a)
    print(b.shape)
