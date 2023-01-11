from PIL import Image
from torchvision import  transforms
from  torch.utils.tensorboard import  SummaryWriter
import numpy as np

#
#

img_path="data1/lianshou/train/ants_image/0013035.jpg"
img_path_abs="C:\\Users\\71530\\PycharmProjects\\pythonProject1\\data1\\lianshou\\train\\ants_image\\0013035.jpg"
img=Image.open(img_path)



writer=SummaryWriter("logs")
# print(tensor_img)
# 如何转化为tensor类型
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)


writer.add_image("Ternsor_img",tensor_img)
writer.close()