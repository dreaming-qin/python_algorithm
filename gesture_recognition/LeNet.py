import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

import numpy as np



class LeNet(nn.Module):

    # 初始化所有的卷积层，池化层和全连接层，还有优化器，损失函数，一些诸如学习率的参数
    def __init__(self):
        # 父类初始化
        super().__init__()
       #定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        self.conv1=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.ReLU()
        )

        # 池化层
        self.poo1=nn.Sequential(
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        )
        self.pool2=nn.Sequential(
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        )

        #最后是三个全连接层
        self.fc1=nn.Sequential(
            nn.Linear(16*5*5,120)
        )
        self.fc2=nn.Sequential(
            nn.Linear(120,84)
        )
        self.fc3=nn.Sequential(
            nn.Linear(84,10)
        )

        # 定义优化器，学习率取0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.99))
        # self.optimizer=optim.SGD(self.parameters(),lr = 0.001,momentum = 0.9)
        # 使用设备：GPU或CPU
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 用交叉熵损失函数为目标函数
        self.loss_fuc=nn.CrossEntropyLoss()
        self.to(self.device) 
    
    # 得到模型输出，输入数据片大小的图片
    def get_output(self,image):
        ans=self.conv1(image)
        ans=self.poo1(ans)
        ans=self.conv2(ans)
        ans=self.pool2(ans)
        ans = ans.view(ans.size(0), -1)
        ans=self.fc1(ans)
        ans=self.fc2(ans)
        ans=self.fc3(ans)
        return ans

    # 训练模型函数，返回一个总损失熵值来打点
    def train_model(self,image,label,cnt):
        self.train()
        # 以ticks张为一次数据片进行传入，获得总的熵值，输出
        images=[]
        labels=[]
        # 数据片
        ticks=2000
        # 总损失熵值
        loss_sum=0
        for i in range(len(image)):            
            images.append([image[i]])
            labels.append(label[i])
            if (i+1)%ticks==0 or i==len(image)-1:
                images=torch.tensor(images).float()
                labels=torch.tensor(labels).long()
                # 有GPU用GPU加速
                images,labels=images.to(self.device),labels.to(self.device)
                # 梯队清零
                self.optimizer.zero_grad()
                output=self.get_output(images)
                # 获得交叉熵的值
                loss=self.loss_fuc(output,labels)
                # 回滚更新权值
                loss.backward()
                self.optimizer.step()
                print("第{}次训练：对第{}张到第{}张图片进行训练后，熵值为{}".format(
                    cnt,(int(i/ticks)*ticks),i,loss.item()
                ))
                loss_sum+=loss.item()
                images=[]
                labels=[]
        return loss_sum

    # 与train_model方法极其类似，不做过多说明
    def test_model(self,image,label,times):
        self.eval()
        cnt=0
        images=[]
        labels=[]
        for i in range(len(image)):
            images.append([image[i]])
            labels.append(label[i])

        images,labels=torch.tensor(images).float(),torch.tensor(labels).long()
        images,labels=images.to(self.device),labels.to(self.device)
        output=self.get_output(images).tolist()
        for i in range(len(output)):
            if label[i]==np.argmax(output[i]):
                cnt=cnt+1
        
        print("第{}次训练的准确率为：{}%".format(times,cnt*100/len(label)))
        return cnt/len(label)