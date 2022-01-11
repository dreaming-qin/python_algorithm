#  -*-coding:utf-8-*-
from typing import Sized
import pandas as pd
import numpy as np
from Bayes import Bayes
from KNN import KNN
from desicion_tree import desicion_tree
from discretization import discretization
from sklearn.tree import DecisionTreeClassifier
import csv
from sklearn.model_selection import train_test_split

from fit import fit
import sys
import time




# 首先，获得拟合后的数据，用线性拟合对缺失值补足
data=fit("..\\iris_nan.csv")
# 将属性和类分开
data=np.array(data)
attr=data[:,0:len(data[0])-1]
target=data[:,len(data[0])-1]
target=target.astype(int)

# 记录各个算法运行的时间和准确率
score_KNN=0
score_tree=0
score_Bayes=0
time_KNN=0
time_tree=0
time_Bayes=0


# 循环100遍，观察结果，取均值
for i in range(100):
    # 进度条代码
    print("\r", end="")
    print("running: {}%: ".format(i), "▋" * (i // 2), end="")
    sys.stdout.flush()

    
    # 获得训练集和测试集
    attr_train, attr_test, target_train, target_test = train_test_split(attr, target, train_size=(2/3))


    start_time=time.time()
    # 连续型数据可以直接使用KNN
    score_KNN=score_KNN+KNN(attr_train, attr_test, target_train, target_test,10)
    end_time=time.time()
    time_KNN=time_KNN+end_time-start_time

    # 离散化
    # 数据为连续型，要离散化，为后面的决策树和贝叶斯分类做准备
    # 这里把它分成3类，正好对应类的三类，使用样本的均值和方差进行分隔，分隔的数据存储在data.csv中
    # 从discretization.csv读数据
    df = pd.read_csv("discretization.csv", encoding='utf-8')
    block=df.iloc[:,:].values.tolist()
    # 离散化函数
    attr_train=np.array(discretization(attr_train,block)).astype(int)
    attr_test=np.array(discretization(attr_test,block)).astype(int)


    start_time=time.time()
    # 贝叶斯朴素算法
    score_Bayes=score_Bayes+Bayes(attr_train, attr_test, target_train, target_test)
    end_time=time.time()
    time_Bayes=time_Bayes+end_time-start_time


    # 决策树算法
    # 开始构建决策树
    d_tree=desicion_tree()
    start_time=time.time()
    d_tree.fit(attr_train,target_train)
    # 可视化决策树，存到tree.png中
    d_tree.print_tree_to_file("tree")
    # 加入测试集，计算准确率
    score_tree=score_tree+d_tree.test(attr_test,target_test)
    end_time=time.time()
    time_tree=time_tree+end_time-start_time

print('')
print('accuracy：')
print('KNN：'+str(score_KNN/100))
print('Bayes：'+str(score_Bayes/100))
print('tree：'+str(score_tree/100))
print('----------------------------------------')
print("running 100 times：")
print('KNN：'+str(time_KNN))
print('Bayes：'+str(time_Bayes))
print('tree：'+str(time_tree))









