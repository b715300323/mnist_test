import  torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear, Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import  model
from  model import *

train_data=torchvision.datasets.MNIST(root="./data" , train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data=torchvision.datasets.MNIST(root="./data", train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size=len(train_data)
test_data_size=len(test_data)
# print("测试数据集长度为:{}".format(train_data_size))
# print("训练数据集长度为:{}".format(test_data_size))


train_dataloader=DataLoader(train_data, batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)



# device = torch.device("cpu")	# 使用cpu训练
# device = torch.device("cuda")	# 使用gpu训练
# device = torch.device("cuda:0")	# 当电脑中有多张显卡时，使用第一张显卡
#device = torch.device("cuda:1")	   # 当电脑中有多张显卡时，使用第二张显卡
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            # nn.Conv2d(3, 32, 5, 1, 2),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 32, 5, 1, 2),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 64, 5, 1, 2),
            # nn.MaxPool2d(2),
            # nn.Flatten(),
            # nn.Linear(64 * 4 * 4, 64),
            # #nn.ReLU(),
            # #nn.Dropout(),
            # nn.Linear(64, 10)
            nn.Conv2d(1,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.model2=nn.Sequential(
            nn.Linear(14 * 14 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.model(x)
        x=x.view(-1,14*14*128)
        x=self.model2(x)
        return x

#创建网络模型
tudui = Tudui()
tudui = tudui.cuda()


#损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()

#优化器
learning_rate=0.01

optimizer=torch.optim.Adam(tudui.parameters(),lr=learning_rate)

#设置训练网络的一个参数
#记录训练次数
total_train_step=0
#记录测试次数
total_test_step=0
#记录测试轮数
epoch=100
#添加tensorboard
writer=SummaryWriter("./logs_handnumber")


for i in range(epoch):
    print("---------第 {} 轮训练开始-------".format(i+1))


    #训练开始
    tudui.train()
    for data in train_dataloader:
        imgs,targets=data
        imgs=imgs.cuda()
        targets=targets.cuda()
        outputs=tudui(imgs)
        loss=loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step=total_train_step+1
        if total_train_step % 100 == 0:
            print("训练次数:{} , Loss：  {}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)


    #测试开始
    tudui.eval()
    total_test_loss=0
    #添加整体正确率
    total_accuracy=0
    with torch.no_grad():
         for data in  test_dataloader:
             imgs,targets=data
             imgs = imgs.cuda()
             targets = targets.cuda()
             outputs = tudui(imgs)
             loss = loss_fn(outputs,targets)
             total_test_loss = total_test_loss+loss.item()
             accuracy = (outputs.argmax(1) == targets).sum()
             total_accuracy = total_accuracy+accuracy

    print("整体测试集上loss: {}".format(total_test_loss))
    print("整体测试集上正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("测试正确率",total_accuracy/test_data_size,total_test_step)
    total_test_step=total_test_step+1

    torch.save(tudui,"handnumber_{}.pth".format(i+1))
    #
    print("模型已保存")

writer.close()