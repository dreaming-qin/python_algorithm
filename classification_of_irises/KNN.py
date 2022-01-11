from queue import PriorityQueue

import numpy as np

# KNN算法
# 返回这个训练集的准确率
def KNN(attr_train, attr_test, target_train, target_test,k=10):
    # 也就是一个个遍历，看一下对不对
    # 距离采用欧氏距离
    # ans是正确的个数
    ans=0
    for i in range(len(attr_test)):
        # 默认小顶堆，反过来存
        q=PriorityQueue()
        q.put((-999999999,1))
        # 遍历训练集，找K个最近邻
        for j in range(len(attr_train)):
            dis=get_distance(attr_test[i],attr_train[j])
            var1=q.get()
            if q.qsize()<k-1:
                q.put(var1)
                q.put((-dis,target_train[j]))
            elif -dis>var1[0]:
                q.put((-dis,target_train[j]))
            else:
                q.put(var1)
        # 统计类型
        dic={}
        while q.qsize()!=0:
            var1=q.get()
            if var1[1] in dic:
                dic[var1[1]]=dic[var1[1]]+1
            else:
                dic[var1[1]]=1
        # 遍历dic
        max_target=0
        var1=0
        for key in dic:
            if dic[key]>var1:
                var1=dic[key]
                max_target=key
        if(max_target==target_test[i]):
            ans=ans+1
    
    return ans/len(target_test)

def get_distance(var1,var2):
    var3=np.subtract(var1,var2)
    return sum(np.multiply(var3,var3))
