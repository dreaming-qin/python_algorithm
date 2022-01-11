import os
import numpy as np
import matplotlib.image

# 从相应的图片，label.txt文件中获得以矩阵表示的image和label数组
# isTrain表示希望获得的数据是否是训练集，true就是获得训练集，false就是获得测试集
def get_data(isTrain=False):
    # 获得相应位置的路径
    var1='test'
    if isTrain:
        var1='train'
    label_path='./data/'+var1+'_label.txt'
    img_path='./data/'+var1+'_pic'

    # 首先，获得标签集
    f=open(label_path)
    # 标签只有一行，只读一行就行
    var1=f.readline()
    label=var1.split(',')
    label=np.delete(label,len(label)-1)
    label=[int(label[i]) for i in range(len(label))]

    # 其次，获得图片矩阵
    image=[]
    for i in range(len(label)):
        im = matplotlib.image.imread(img_path+'/'+str(i)+'.jpg')
        im=np.divide(im,255)
        image.append(im)

    return image,label

