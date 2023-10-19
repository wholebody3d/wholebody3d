import os
import torch
import torch.nn as nn
import torchvision.models as models

class model_resnet50(nn.Module):
    def __init__(self, num_keypoint=133, pretrained=False):
        super(model_resnet50, self).__init__()
        self.encoder = models.resnet50(pretrained=pretrained)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7 * 7 * 512 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(7 * 7 * 512, 512)
        # self.fc4 = nn.Linear(512, 512)
        self.outlayer1 = nn.Linear(1024, num_keypoint*3)
        # self.outlayer2 = nn.Linear(512, num_keypoint*2)

    def forward(self, x):
        x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)  # 100352 = 28 x 28 x 128
        x = self.encoder.layer3(x)  # 50176 = 14 x 14 x 256
        x = self.encoder.layer4(x)  # 25088 = 7 x 7 x 512 # 100352
        x = x.reshape(x.shape[0], -1)
        x1 = self.relu(self.fc1(x))
        x1 = self.relu(self.fc2(x1))
        x1 = (self.outlayer1(x1))
        # x2 = self.relu(self.fc3(x))
        # x2 = self.relu(self.fc4(x2))
        # x2 = (self.outlayer2(x2))
        return x1 #, x2



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model_resnet50(pretrained=False).to(device)
    if os.path.exists('./net.pth'):
        net.load_state_dict(torch.load('./net.pth', map_location=device))
        print('load pretrained weight')

    batch_size = 2
    a = torch.rand(batch_size, 3, 224, 224).to(device)
    b = net(a)
    print(b.shape)


