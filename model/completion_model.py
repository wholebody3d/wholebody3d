from __future__ import absolute_import, division, print_function

import math
import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CompletionTransformer(nn.Module):
    def __init__(self, num_keypoints=133, dim_input=3, dim_output=3):
        super(CompletionTransformer, self).__init__()
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

