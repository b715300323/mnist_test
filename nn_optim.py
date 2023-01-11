import  torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear, Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=1)



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

#交叉熵损失函数
loss=nn.CrossEntropyLoss()

tudui=Tudui()
#优化器
optim=torch.optim.SGD(tudui.parameters(),lr=0.0015)

for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        outputs=tudui(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        # print(outputs)
        # print(targets)
        # print(result_loss)
        running_loss=running_loss+result_loss
    print(running_loss)
