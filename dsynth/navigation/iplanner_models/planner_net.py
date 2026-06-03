import torch
import torch.nn as nn
from .percept_net import PerceptNet


class PlannerNet(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        self.encoder = PerceptNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x, goal):
        x = self.encoder(x)
        x, c = self.decoder(x, goal)
        return x, c


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(256 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k * 3)
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, goal), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        f = self.relu(self.fc1(x))
        x = self.relu(self.fc2(f))
        x = self.fc3(x)
        x = x.reshape(-1, self.k, 3)
        c = self.relu(self.frc1(f))
        c = self.sigmoid(self.frc2(c))
        return x, c
