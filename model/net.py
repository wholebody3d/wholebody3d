from __future__ import absolute_import, division, print_function

import math
import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class net_transformer_completion(nn.Module):
    def __init__(self, num_keypoints=133, dim_input=3, dim_output=3):
        super(net_transformer_completion, self).__init__()
        self.dim_input=dim_input
        self.num_position_encoding = 16
        self.inlayer = nn.Linear((self.num_position_encoding) * self.dim_input, 64)
        encoderlayer = TransformerEncoderLayer(d_model=64, nhead=1,dim_feedforward=64, dropout=0.0, activation="relu", batch_first=True)
        self.tranformerencoder1 = TransformerEncoder(encoderlayer, num_layers=2)
        self.tranformerencoder2 = TransformerEncoder(encoderlayer, num_layers=2)
        self.tranformerencoder3 = TransformerEncoder(encoderlayer, num_layers=2)
        self.tranformerencoder4 = TransformerEncoder(encoderlayer, num_layers=2)
        self.outlayer = nn.Linear(64, dim_output)
        self.learned_position = nn.Parameter(torch.randn(1,num_keypoints, self.num_position_encoding*self.dim_input))

    def forward(self, x, currimask, output_intermediate=False, final_multiplier=1000):
        x, mask = self.preprocess(x)
        batch_size, _, _ = x.shape
        x = self.inlayer(x)

        cmask = (currimask[1][:,:,0].unsqueeze(1).repeat_interleave(133, dim=1).repeat_interleave(batch_size, dim=0))
        x = self.tranformerencoder1(x, mask=cmask * 999.0 - 999.0)
        x1 = self.outlayer(x) * final_multiplier

        cmask = (currimask[2][:,:,0].unsqueeze(1).repeat_interleave(133, dim=1).repeat_interleave(batch_size, dim=0))
        x = self.tranformerencoder2(x, mask= cmask * 999.0 - 999.0)
        x2 = self.outlayer(x) * final_multiplier

        cmask = (currimask[3][:, :, 0].unsqueeze(1).repeat_interleave(133, dim=1).repeat_interleave(batch_size, dim=0))
        x = self.tranformerencoder3(x, mask= cmask * 999.0 - 999.0)
        x3 = self.outlayer(x) * final_multiplier

        cmask = (currimask[4][:, :, 0].unsqueeze(1).repeat_interleave(133, dim=1).repeat_interleave(batch_size, dim=0))
        x = self.tranformerencoder4(x, mask= cmask * 999.0 - 999.0)
        x4 = self.outlayer(x) * final_multiplier

        if output_intermediate:
            return [x1, x2, x3, x4]
        else:
            return x4

    def preprocess(self, x):
        batch_size, _, _ = x.shape
        mask = (x[:, :, -1] > 0.0001) * 1.0
        mask = (mask.unsqueeze(1).repeat_interleave(133, dim=1))
        x = x[:, :, :self.dim_input]
        x = torch.repeat_interleave(x, self.num_position_encoding, dim=2)
        for j in range(self.dim_input):
            for i in range(self.num_position_encoding):
                x[..., i + j * self.num_position_encoding] = torch.cos(x[..., i + j * self.num_position_encoding] / 2000 * math.pi * (2.0 ** (i // 2)) + (i % 2 == 0) * math.pi / 2.0)
        x = x + self.learned_position
        return x, mask


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



