import numpy as np
import math
import itertools
from scipy.special import comb

def global_(v1,v2):
    vector_a = np.mat(v1)
    vector_b = np.mat(v2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos

def part(v1,v2):
    # 分子
    sum_score=0
    # 循环次数
    round=3
    for i in range(round):
        list1=itertools.combinations(v1, 2+i)
        items1=[0 for i in range(int(comb(len(v1),2+i)))]
        var2=0
        for var1 in list1:
            items1[var2]=var1
            var2=var2+1
        list2=itertools.combinations(v2, 2+i)
        items2=[0 for i in range(int(comb(len(v2),2+i)))]
        var2=0
        for var1 in list2:
            items2[var2]=var1
            var2=var2+1

        var1=0
        for i in range(len(items2)):
            if np.count_nonzero(items1[i])==0 or np.count_nonzero(items2[i])==0:
                continue
            var1=var1+global_(items1[i],items2[i])
        sum_score=sum_score+var1/len(items2)
            
    return sum_score/round

#获取想获得的流派数据集
def get_relation_data(src_data,relation_name):
    data_relation=[]
    for item in src_data:
        if relation_name in item[len(item)-1]:
            var1=np.delete(item,len(item)-1,0)
            data_relation.append(var1)
    return data_relation

def get_attr_similarity(data,data_name,sttr_index_list):
    # 存Spart
    Spart=[0 for i in range(len(data)-1)]
    # 开始计算，主要思想是用质心，算两遍，记得除2，就能得到一个Spart_i
    data1=np.array(data[data_name])
    data1=data1[:,sttr_index_list]

    Spart_index=0
    for key in data:
        if key==data_name:
            continue
        data2=np.array(data[key])
        data2=data2[:,sttr_index_list]
        
        # 获得data1的质心
        mean_data1=data1.mean(axis=0)
        mean_data2=data2.mean(axis=0)

        # 先计算data1各点到data2质心的距离，然后平均
        var1=0
        for item in data1:
            var1=var1+global_(mean_data2,item)
            if np.count_nonzero(mean_data2)==0 or np.count_nonzero(item)==0:
                return [np.nan]
        var1=var1/len(data1)
        # 再计算data2各点到data1质心的距离，然后平均
        var2=0
        for item in data2:
            var2=var2+global_(mean_data1,item)
            if np.count_nonzero(mean_data1)==0 or np.count_nonzero(item)==0:
                return [np.nan]
        var2=var2/len(data2)

        # 赋值
        Spart[Spart_index]=(var1+var2)/2
        Spart_index=Spart_index+1

    return np.array(Spart)

        
