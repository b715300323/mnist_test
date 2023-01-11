import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

#创立数据
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32)
#池化要求数据为浮点数，需转化数据类型
input=torch.reshape(input,(-1,1,5,5))
#—  -1代表自动识别channel
print(input.shape)

#创立池化核
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

#ceil_mode代表对于数据无法完整匹配池化核的情况是否依然计算，true为是，false则舍弃数据

    def forward(self,input):
        output=self.maxpool1(input)
        return  output




tudui=Tudui()
# output=tudui(input)
# print(output)

writer=SummaryWriter("logs_maxpool")
step=0

for data in dataloader:
    imgs,targert=data
    writer.add_images("input",imgs,step)
    output=tudui(imgs)
    writer.add_images("output",output,step)
    step=step+1




writer.close()