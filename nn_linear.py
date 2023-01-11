import  torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True )


dataloader=DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        #线性变换
        self.linear1=Linear(196608,1)

    def  forward(self,input):
        output=self.linear1(input)
        return  output

tudui=Tudui()

for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    #将数据转变为一维
    output=torch.flatten(imgs)
    print(output.shape)
    output=tudui(output)
    print(output.shape)