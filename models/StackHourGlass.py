import os
import torch
import torch.nn as nn

def batchnorm_shg(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv_shg(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv_shg, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual_shg(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual_shg, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv_shg(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv_shg(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv_shg(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv_shg(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class Hourglass_shg(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass_shg, self).__init__()
        nf = f + increase
        self.up1 = Residual_shg(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual_shg(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass_shg(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual_shg(nf, nf)
        self.low3 = Residual_shg(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2

class UnFlatten_shg(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge_shg(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge_shg, self).__init__()
        self.conv = Conv_shg(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class model_PoseNet_shg(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(model_PoseNet_shg, self).__init__()

        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv_shg(3, 64, 7, 2, bn=True, relu=True),
            Residual_shg(64, 128),
            nn.MaxPool2d(2, 2),
            Residual_shg(128, 128),
            Residual_shg(128, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass_shg(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual_shg(inp_dim, inp_dim),
                Conv_shg(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([Conv_shg(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge_shg(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge_shg(oup_dim, inp_dim) for i in range(nstack - 1)])
        self.nstack = nstack
        # self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        # x = imgs.permute(0, 3, 1, 2)  # x of size 1,3,inpdim,inpdim
        x = self.pre(imgs)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    '''def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:, i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss'''


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model_PoseNet_shg(nstack=4, inp_dim=3, oup_dim=133, bn=False, increase=0).to(device)
    if os.path.exists('./net.pth'):
        net.load_state_dict(torch.load('./net.pth', map_location=device))
        print('load pretrained weight')

    batch_size = 2
    a = torch.rand(batch_size, 3, 256, 256).to(device)
    b = net(a)
    print(b.shape)


