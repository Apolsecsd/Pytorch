from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter('logs')
image_path = "C:/Users/LENOVO/Desktop/jt.png"
image_PIL = Image.open(image_path)
img_array = np.array(image_PIL)
print(type(img_array))
writer.add_image('test',img_array,2,dataformats='HWC')
for i in range(100) :
    writer.add_scalar('y=2x',i, 2*i)    #利用循环来模拟训练过程，将迭代次数（i）作为x轴的值，并将当前迭代次数（i）作为y轴的值
writer.close()
#在pycharm终端上面输入tensorboard --logdir=logs得到网址，打开网址就可获得图像。

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
img_path = 'C:/Users/LENOVO/Desktop/jt.png'
img = Image.open(img_path)    #使用PIL库的open()函数加载指定路径下的图像，并将其赋值给变量img
writer = SummaryWriter('logs')
tensor_trans = transforms.ToTensor()     #使用transforms.ToTensor()将图像转换为张量数据，并将结果赋值给变量tensor_img。
tensor_img = tensor_trans(img)
writer.add_image('tensor_img',tensor_img)
print(tensor_img)

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')
img = Image.open('C:/Users/LENOVO/Desktop/jt.png')
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
img_tensor = img_tensor[:3]  #通过切片操作 [:3] 保留了前三个通道（R、G、B），将结果存储在变量 img_tensor 中
writer.add_image('Totensor',img_tensor)      #使用 add_image() 方法将处理后的图像数据添加到 TensorBoard 日志中，其中 'Totensor' 是记录的名称
print(img_tensor[0][0][0])    #打印输出图像张量的第一个像素的值。
trans_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))     #使用 Normalize() 方法对图像进行归一化处理，将像素值映射到均值为 0.5，标准差为 0.5 的分布
#trans_norm = transforms.Normalize(mean, std)mean 和 std 分别表示图像的均值和标准差。这两个参数需要根据实际应用情况进行设置，可以通过统计大量图像数据集的像素值来得到。
#对于每个通道的像素值，Normalize 的处理方式为 (input[channel] - mean[channel]) / std[channel]。即用图像的像素值减去均值，并除以标准差，最终得到归一化后的像素值。
print(img_tensor.shape)
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize',img_norm)
writer.close()

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')
img = Image.open('C:/Users/LENOVO/Desktop/jt.png')
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
img_tensor = img_tensor[:3]  # 保留前三个通道 (R、G、B)
writer.add_image('Totensor',img_tensor)
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize((6,3,2),(9,3,5))   #使用 Normalize() 方法对图像进行归一化处理，将像素值减去均值 (6, 3, 2)，再除以标准差 (9, 3, 5)
print(img_tensor.shape)
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize',img_norm,2)     #2作为全局步数
print(img.size)
trans_resize = transforms.Resize((512,512))     #使用 Resize() 方法对图像进行大小调整，将图像调整为 (512, 512) 的尺寸。
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image('Resize',img_resize,0)
print(img_resize)
trans_resize_2 = transforms.Resize(512)    #将图像调整为宽度和高度都是 512 的尺寸
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('Resize',img_resize_2,1)
writer.close()

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')
img = Image.open('C:/Users/LENOVO/Desktop/jt.png')
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
img_tensor = img_tensor[:3]  # 保留前三个通道 (R、G、B)
writer.add_image('Totensor',img_tensor)
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize((6,3,2),(9,3,5))
print(img_tensor.shape)
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize',img_norm,2)
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image('Resize',img_resize,0)
print(img_resize)
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('Resize',img_resize_2,1)
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range (10) :
    img_crop = trans_compose_2(img)
    writer.add_image('Randomcrop',img_crop,i)
writer.close()