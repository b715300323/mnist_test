# 搭建神经网络
from torch import nn
import  torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear, Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import  model

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '_main_':
    tudui=Tudui()
    input=torch.ones(64,3,32,32)
    output=tudui(input)
    print(output.shape)

