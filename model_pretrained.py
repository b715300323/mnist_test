import  torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear, Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data=torchvision.datasets.ImageNet("./data",split="train",  )