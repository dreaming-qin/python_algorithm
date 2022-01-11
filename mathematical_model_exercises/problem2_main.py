import pandas as pd
import numpy as np
import math
import sys
import random

from MS import get_relation_data, global_, part

filename='../2021_ICM_Problem_D_Data/full_music_data.csv'
df = pd.read_csv(filename)
data = np.array(df.iloc[:,:].values)

filename='../2021_ICM_Problem_D_Data/influence_data.csv'
df = pd.read_csv(filename)
var1 = np.array(df.iloc[:,:].values)
id_to_relation={}
# 通过var1获得一个字典，映射歌手id和所属流派的关系
for item in var1:
    if str(item[0]) not in id_to_relation:
        id_to_relation[str(item[0])]=str(item[2])
    if str(item[4]) not in id_to_relation:
        id_to_relation[str(item[4])]=str(item[6])

# 通过字典给data添加一列，代表流派
var1=[]
for i in range(len(data)):
    item=data[i]
    str1=''
    id_str=item[1]
    id_str=item[1][1:len(id_str)-1].split(', ')
    for id_str_item in id_str:
        if id_str_item in id_to_relation:
            str1=str1+id_to_relation[id_str_item]+' '
    if ''==str1:
        var1.append(i)
    data[i][18]=str1
data=np.delete(data,var1,0)
var1=[0,1,13,15,16,17]
data=np.delete(data,var1,1)

# 对data所有元素归一化
data=np.transpose(data)
var1=[data[i]/np.sum(data[i]) for i in range(len(data)-1)]
var1.append(data[len(data)-1])
data=np.transpose(np.array(var1))

lambda_=0.3
for acc in range(11):
    lambda_=acc/10
    print('lambada为{}'.format(lambda_))
    # 先计算流派内的相似度。使用方法是对质心求score
    str1='Pop/Rock'
    data_relation=get_relation_data(data,str1)

    # 质心
    data_relation=np.transpose(data_relation)
    mean_=[np.sum(data_relation[i])/len(data_relation[i]) for i in range(len(data_relation))]
    mean_=np.array(mean_)
    data_relation=np.transpose(data_relation)
    in_score=0
    stop=min(len(data_relation),100)
    data_relation=random.sample(list(data_relation), stop)
    for i in range(stop):
        # 进度条代码
        print("\r", end="")
        print("running: {:.2f}%: ".format(i*100/stop), end="")
        sys.stdout.flush()

        item=data_relation[i]
        in_score=in_score+lambda_*global_(item,mean_)+(1-lambda_)*part(item,mean_)
        if np.isnan(in_score):
            in_score=in_score+lambda_*global_(item,mean_)+(1-lambda_)*part(item,mean_)
    in_score=in_score/stop
    print('{}的分数是{}'.format(str1,in_score))

    # 计算流派间的，找n个点，取平均值
    str1='Pop/Rock'
    data_relation=get_relation_data(data,str1)
    stop=min(int(math.sqrt(len(data_relation))),10)
    data1=random.sample(list(data_relation), stop)

    str2='Jazz'
    data_relation=get_relation_data(data,str2)
    stop=min(int(math.sqrt(len(data_relation))),10)
    data2=random.sample(list(data_relation), stop)

    between_score=0
    for i in range(len(data1)):
        for j in range(len(data2)):
            # 进度条代码
            print("\r", end="")
            print("running: {:.2f}%: ".format((i*len(data1)+j)*100/(len(data2)*len(data1))), end="")
            sys.stdout.flush()

            between_score=between_score+lambda_*global_(data1[i],data2[j])+(1-lambda_)*part(data1[i],data2[j])
    between_score=between_score/(len(data2)*len(data1))
    print('{}与{}之间的的分数是{}'.format(str1,str2,between_score))
    print('----------------------------------------------------------')

# # 质心
# data_relation=np.transpose(data_relation)
# mean_1=[np.sum(data_relation[i])/len(data_relation[i]) for i in range(len(data_relation))]
# mean_1=np.array(mean_1)
# str2='Jazz'
# data_relation=get_relation_data(data,str2)
# # 质心
# data_relation=np.transpose(data_relation)
# mean_2=[np.sum(data_relation[i])/len(data_relation[i]) for i in range(len(data_relation))]
# mean_2=np.array(mean_2)
# between_score=lambda_*global_(mean_1,mean_2)+(1-lambda_)*part(mean_1,mean_2)
# print('{}与{}之间的的分数是{}'.format(str1,str2,between_score))