# 朴素贝叶斯分类算法

# 返回正确率
import numpy as np


def Bayes(attr_train, attr_test, target_train, target_test):
    # 在这里对训练集按类先排好序，为后续做准备
    data=attr_train.tolist()
    for i in range(len(attr_train)):
        data[i].append(target_train[i])
    var1=len(data[0])-1
    data.sort(key=lambda data: data[var1], reverse=False)
    data=np.array(data)

    attr_train=data[:,0:len(data[0])-1]
    target_train=data[:,len(data[0])-1]

    ans=0
    # 存答案，因为属性最多就81种可能，与其每次遍历，还不如把答案存下来
    save={}
    for i in range(len(target_test)):
        str1=str(attr_test[i])
        flag=False
        if str1 in save:
            flag=save[str1]
        else:
            var1=Bayes_a_transaction(attr_train, attr_test[i], target_train)
            flag=(var1==target_test[i])
            save[str1]=flag
        if flag:
            ans=ans+1
    
    return ans/len(target_test)

# 返回一个事务预测分类
def Bayes_a_transaction(attr_train, transaction, target_train):
    # 存答案，key是类，value是这个类的概率
    ans={}
    s_index = 0
    for i in range(0, len(target_train)):
        if i == len(target_train)-1 or target_train[i] != target_train[i+1]:
            var1 = Bayes_a_transaction_by_class(attr_train[s_index:i+1,:], transaction,(i-s_index+1)/len(target_train))
            ans.update({target_train[i]:var1})
            s_index = i+1

    # 找到最大概率的类
    max_p=0
    max_class=0
    for key in ans:
        if ans[key]>max_p:
            max_p=ans[key]
            max_class=key

    return max_class

# 返回这个事务是这个类的概率
# pY是P（target）
def Bayes_a_transaction_by_class(attr_train, transaction, pY):
    # 存在attr_train中属性ai和transaction相同的数量，key是ai，value是数量
    cnt={}
    # 防止过拟合
    for i in range(len(transaction)):
        cnt[i]=1
    for i in range(len(attr_train)):
        for j in range(len(attr_train[i])):
            if attr_train[i][j]==transaction[j]:
                cnt[j]=cnt[j]+1
    
    ans=1
    for key in cnt:
        ans=ans*(cnt[key]/len(attr_train))*10
    ans=ans*pY*10
    return ans