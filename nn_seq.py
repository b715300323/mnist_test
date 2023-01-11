import  torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear, Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()


        self.model1=nn.Sequential(Conv2d(3,32,5,padding=2),
                                  MaxPool2d(2),
                                  Conv2d(32,32,4,padding=2),
                                  MaxPool2d(2),
                                  Conv2d(32,64,5,padding=2),
                                  MaxPool2d(2),
                                  Flatten(),
                                  Linear(1024, 64),
                                  Linear(64,10))


    def forward(self,x):
       x=self.model1(x)
       return x

tudui=Tudui()
print(tudui)

input=torch.ones([64,3,32,32])
output=tudui(input)
print(output.shape)

writer=SummaryWriter("logs_seq")
writer.add_graph(tudui,input)

x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])















writer.close()