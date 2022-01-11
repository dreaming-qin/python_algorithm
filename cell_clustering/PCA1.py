
import numpy as np
from numpy.lib.function_base import append

# 传入源数据，源数据是n×m的数据矩阵，n是事务个数，m是事务特征个数
# k是降维后的维度，当k=-1是默认值,k=其它值时代表它传入了这个参数，否则k值由程序决定
# 返回降维后的数据，它是n×k的矩阵
def PCA(src_data,k=-1):
    # 求均值
    avg = np.array([np.mean(src_data[:, i]) for i in range(len(src_data[0]))])
    src_sub=src_data-avg
    print(1)
    # 求协方差矩阵
    C=np.dot(np.transpose(src_sub),src_sub)
    print(2)
    # C=np.divide(C,len(src_data))
    # 求特征值，特征向量
    val, vec = np.linalg.eig(C)
    print(3)
    # 将特征值和特征向量组合
    characteristic=[]
    for i in range(len(val)):
        characteristic.append([val[i],vec[:,i]])
    print(4)
    # 然后排序
    characteristic.sort(key=lambda characteristic: characteristic[0], reverse=True)
    print(5)
    # 当k=-1时，意味着我们要自己决定k
    if k==-1:
        # stop为阈值
        stop=0.95
        sum=0
        for i in range(len(characteristic)):
            sum=sum+characteristic[i][0]
        stop=stop*sum
        for i in range(len(characteristic)-1,-1,-1):
            if sum-characteristic[i][0]>=stop:
                sum=sum-characteristic[i][0]
            else:
                break
        k=i+1
        
    # 获得前k个特征向量，组成矩阵
    vec=[]
    for i in range(k):
        vec.append(characteristic[i][1])

    # 与减去平均值的矩阵相乘，获得降维后的数据
    data=np.dot(src_sub,np.transpose(vec))
    return data