
import torch
import torchvision
from  torchvision import datasets,transforms
from torch .autograd import  Variable

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)
