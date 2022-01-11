import matplotlib.pyplot as plt
import torch
from LeNet import LeNet
from data_visual import data_visual
from get_data import get_data


# 先获得图片形式的数据和标签
data_visual()

# 实例化一个LeNet5模型
model=LeNet()
# 加载模型
model.load_state_dict(torch.load("./result/model"))
# 得到训练集和测试集
train_image,train_label=get_data(isTrain=True)
test_image,test_label=get_data(isTrain=False)
# 画图，将点用列表存储
train_loss=[[],[]]
test_accurancy=[[],[]]
# 训练100回
for i in range(100):
    var1=model.train_model(train_image,train_label,i)
    train_loss[0].append(i)
    train_loss[1].append(var1)

    # 测试
    cnt=model.test_model(test_image,test_label,i)
    test_accurancy[0].append(i)
    test_accurancy[1].append(cnt)

# 画图1
plt.plot(train_loss[0],train_loss[1], color='b')
plt.xlabel("cnt")
plt.ylabel("sum of loss")
plt.savefig('./result/train_loss.png')
plt.close()
# 画图2
plt.plot(test_accurancy[0],test_accurancy[1], color='b')
plt.xlabel("cnt")
plt.ylabel("accurancy")
plt.savefig('./result/test_accurancy.png')
plt.close()

# 保存模型
torch.save(model.state_dict(),"./result/model")