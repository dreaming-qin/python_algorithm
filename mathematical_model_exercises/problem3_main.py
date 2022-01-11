import pandas as pd
import numpy as np
import math
import sys
import random
import itertools
from scipy.special import comb

from MS import get_attr_similarity, get_relation_data, global_


filename='../2021_ICM_Problem_D_Data/data_by_artist.csv'
df = pd.read_csv(filename)
data = np.array(df.iloc[:,:].values)
index_to_attr_dict={}
var1=np.array(df.columns)
var1=np.delete(var1,[0,1,7,8,14,15],0)
for i in range(len(var1)):
    index_to_attr_dict[i]=var1[i]

filename='../2021_ICM_Problem_D_Data/influence_data.csv'
df = pd.read_csv(filename)
var1 = np.array(df.iloc[:,:].values)
id_to_relation={}
relation_set=set()
# 通过var1获得一个字典，映射歌手id和所属流派的关系
for item in var1:
    if str(item[0]) not in id_to_relation:
        id_to_relation[str(item[0])]=str(item[2])
        relation_set.add(str(item[2]))
    if str(item[4]) not in id_to_relation:
        id_to_relation[str(item[4])]=str(item[6])
        relation_set.add(str(item[6]))


# 通过字典给data添加一列，代表流派
var1=[]
for i in range(len(data)):
    item=data[i]
    str1=''
    id_str=str(item[1])
    if id_str in id_to_relation:
        str1=str1+id_to_relation[id_str]
    else:
        var1.append(i)
    data[i][0]=str1
data=np.delete(data,var1,0)
var1=[1,7,8,14,15]
data=np.delete(data,var1,1)

# 对data所有元素归一化
data=np.transpose(data)
var1=[data[i]/np.sum(data[i]) for i in range(1,len(data))]
var1.append(data[0])
data=np.transpose(np.array(var1))

# 将数据分组，存到三维数组中
var1={}
relation_set.remove('Unknown')
for key in relation_set:
    var1[key]=get_relation_data(data,key)
data=var1

# 获得所有组合
var1=[i for i in range(len(var1[key][0]))]
round=5
var2=0
for i in range(round):
    var2=var2+int(comb(len(var1),2+i))
group=[0 for i in range(var2)]
var2=0
for i in range(3):
    list1=itertools.combinations(var1, 2+i)
    for item in list1:
        group[var2]=item
        var2=var2+1
group=np.array(group)


# 获得结果
sigma=0.7
ans='genre,attr,score\n'
# get_ans_data_name_set=set(['Latin','Electronic'])
get_ans_data_name_set=set(relation_set)
get_ans_data_name_set.remove('Pop/Rock')
for data_name in get_ans_data_name_set:
    Spart=[0 for i in range(len(group))]
    discribe=[0 for i in range(len(group))]
    # 记录Spart是nan的列表
    var1=[]
    # 先获得Spart
    for i in range(len(group)):
        # 进度条代码
        print("\r", end="")
        print("running {} : {:.2f}%: ".format(data_name,i*100/len(group)), end="")
        sys.stdout.flush()
        
        Spart[i]=np.array(get_attr_similarity(data,data_name,group[i]))
        if np.isnan(Spart[i]).any():
            var1.append(i)
    Spart=np.array(np.delete(Spart,var1,0)).tolist()
    discribe=np.array(np.delete(discribe,var1,0)).tolist()

    # 获得每个属性之中的最小值
    min_Spart=np.array(np.min(Spart, axis=0))
    # 开始discribe，存在discribe中
    for i in range(len(Spart)):
        # 欧氏距离
        Se=math.sqrt(np.sum((Spart[i]-min_Spart)**2))
        Se=1/(1+Se)
        # 余弦相似度
        Scos=global_(Spart[i],min_Spart)
        score=sigma*Se+(1-sigma)*Scos
        discribe[i]=[score,i]
    # 排序
    discribe.sort(key=lambda discribe: discribe[0], reverse=True)
    # 选取最优的前round名
    round=10
    for i in range(round):
        var1=group[discribe[i][1]]
        str1=''
        for j in range(len(var1)-1):
            str1=str1+index_to_attr_dict[var1[j]]+' and '
        str1=str1+index_to_attr_dict[var1[len(var1)-1]]
        ans=ans+'{},{},{}\n'.format(data_name,str1,discribe[i][0])

    print('\n----------------------------------------')

print(ans)
file=open('data/problem3.csv','w',errors='ignore')
file.write(ans)
file.close()

    


