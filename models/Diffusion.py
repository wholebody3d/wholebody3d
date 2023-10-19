from __future__ import absolute_import, division, print_function

import os
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class res_block(nn.Module):
    def __init__(self, hidden=1024):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp

        return x


class BodyPartCorrector(nn.Module):
    def __init__(self, num_keypoint=21, num_out=42, pretrained=False, combine=False):
        super(BodyPartCorrector, self).__init__()
        self.encoder = models.resnet18(pretrained=pretrained)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7 * 7 * 512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.outlayer = nn.Linear(256, num_out)

        self.fc_in = nn.Linear(num_keypoint * 2, 256)
        self.fc_1 = res_block(256)
        self.fc_2 = res_block(256)
        self.fc_3 = res_block(256)
        self.fc_out = nn.Linear(256, num_out)
        self.combine = combine
        if self.combine:
            self.fc_final_1 = nn.Linear(num_out*2, 256)
            self.fc_final_2 = res_block(256)
            self.fc_final_3 = res_block(256)
            self.fc_final_4 = nn.Linear(256, num_out)
            self.fc_final_5 = nn.Linear(num_out, 256)
            self.fc_final_6 = nn.Linear(256, 1)

    def forward_image(self, x):
        x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)  # 100352 = 28 x 28 x 128
        x = self.encoder.layer3(x)  # 50176 = 14 x 14 x 256
        x = self.encoder.layer4(x)  # 25088 = 7 x 7 x 512
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = (self.outlayer(x))
        return x

    def forward_tensor(self, y):
        y = nn.LeakyReLU()(self.fc_in(y))
        y = nn.LeakyReLU()(self.fc_1(y))
        y = nn.LeakyReLU()(self.fc_2(y))
        y = nn.LeakyReLU()(self.fc_3(y))
        y = (self.fc_out(y))
        return y

    def forward(self, x, y, given_x=False, given_y=False):
        x0 = x
        if not given_x:
            x0 = self.forward_image(x0)
        y0 = y
        if not given_y:
            y0 = self.forward_tensor(y0)

        if self.combine:
            x0 = torch.cat((x0, y0), dim=-1)
            x0 = nn.LeakyReLU()(self.fc_final_1(x0))
            x0 = nn.LeakyReLU()(self.fc_final_2(x0))
            x0 = nn.LeakyReLU()(self.fc_final_3(x0))
            x0 = nn.Sigmoid()(self.fc_final_4(x0))
            v0 = nn.LeakyReLU()(self.fc_final_5(x0))
            v0 = nn.Sigmoid()(self.fc_final_6(v0))
            return [x0, v0]
        else:
            return [x0, y0]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = BodyPartCorrector(21, 21*2, combine=True).to(device)
    if os.path.exists('./net_hand.pth'):
        net.load_state_dict(torch.load('./net_hand.pth', map_location=device))
        print('load pretrained hand weight')

    batch_size = 2
    a = torch.rand(batch_size, 21*2).to(device)
    img = torch.rand(batch_size, 3, 224, 224)
    b = net(img, a)
    print(b[0].shape)
    print(b[1].shape)

    net = BodyPartCorrector(68, 68 * 2, combine=True).to(device)
    if os.path.exists('./net_face.pth'):
        net.load_state_dict(torch.load('./net_face.pth', map_location=device))
        print('load pretrained hand weight')

    batch_size = 2
    a = torch.rand(batch_size, 68 * 2).to(device)
    img = torch.rand(batch_size, 3, 224, 224)
    b = net(img, a)
    print(b[0].shape)
    print(b[1].shape)



