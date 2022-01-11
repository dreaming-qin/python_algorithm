import torchvision
import os

# 通过压缩包格式获得训练集和测试集的图片和相应标签
# 训练集图片在data\train_pic中，标签在data\train_label.txt
# 测试集图片在data\test_pic中，标签在data\test_label.txt
# 其中，标签数据用‘,’分隔，图片以i.jpg的方式进行存储，它代表标签文件的第i个数据就是这张图片的标签
def data_visual():
    mnist_train=torchvision.datasets.MNIST('./data',train=True,download=True)#首先下载数据集，并数据分割成训练集与数据集
    mnist_test=torchvision.datasets.MNIST('./data',train=False,download=True)
    
    # 生成存放图片的文件夹
    if not os.path.exists('./data/train_pic'):
        os.makedirs('./data/train_pic')
    if not os.path.exists('./data/test_pic'):
        os.makedirs('./data/test_pic')
    
    f=open("./data/train_label.txt",'w')#在指定路径之下生成.txt文件
    for i,(img,label) in enumerate(mnist_train):
        img_path = "./data/train_pic"+"/" + str(i) + ".jpg"
        img.save(img_path)
        f.write(str(label)+',')#将路径与标签组合成的字符串存在.txt文件下
    f.close()#关闭文件

    f=open("./data/test_label.txt",'w')#在指定路径之下生成.txt文件
    for i,(img,label) in enumerate(mnist_test):
        img_path = "./data/test_pic"+"/" + str(i) + ".jpg"
        img.save(img_path)
        f.write(str(label)+',')#将路径与标签组合成的字符串存在.txt文件下
    f.close()#关闭文件