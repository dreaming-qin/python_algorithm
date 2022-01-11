import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# 这个函数用于线性拟合，填补缺失数据
def fit(filename):

    # 获得数据
    # filename = "C:\\Users\\86186\\Desktop\\作业\\大三上\\数据挖掘\\实验一\\iris_nan.csv"
    df = pd.read_csv(filename, encoding='gbk')
    # value是属性，target是目标
    value = np.array(df.iloc[:, 0:4].values)
    target = np.array(df.iloc[:, 4].values)

    # 获得那些需要填补数据的所在的列，把它们分离出来，并保存在miss_value_attr和miss_value_target中
    nan_rows = df[df.isnull().T.any().T].T.columns.asi8
    miss_value_attr = value[nan_rows].tolist()
    miss_value_target = target[nan_rows].tolist()
    # 然后还要在原数据删除这些nan的行
    value = np.delete(value, nan_rows, 0)
    target = np.delete(target, nan_rows, 0)

    # 对value和target根据类进行分组，然后直接拟合
    # 答案存在ans中
    ans = []
    var1 = value
    var2 = target
    s_index = 1
    e_index = 0
    for i in range(0, var2[len(var2)-1]+1):
        # 根据类进行分组
        value = []
        target = []
        for e_index in range(s_index, len(var1)+1):
            if e_index==len(var1) or var2[e_index] != var2[e_index-1]:
                break
        value = var1[s_index-1:e_index, :]
        target = var2[s_index-1:e_index]
        s_index = e_index+1
        # 做好准备工作，开始拟合
        cft = linear_model.LinearRegression()
        for j in range(0, len(value[0])):
            # 得到每个属性关于第i个属性的相关线性关系
            var3 = np.delete(value, j, 1)
            var4 = value[:, j]
            cft.fit(var3, var4)
            # 得到后，遍历miss_value_attr数组，对nan数值进行预测
            for k in range(0, len(miss_value_attr)):
                if np.isnan(miss_value_attr[k][j]) and miss_value_target[k] == target[j]:
                    var3 = np.delete(miss_value_attr[k], j, 0).reshape(1, -1)
                    var4 = cft.predict(var3)
                    miss_value_attr[k][j] = round(var4[0], 1)
                    # 拟合完成后，存到ans中
                    miss_value_attr[k].append(miss_value_target[k])
                    ans.append(miss_value_attr[k])
        # 将value和target数据放到ans中
        var3 = value.tolist()
        var4 = target.tolist()
        for j in range(0, len(value)):
            var3[j].append(var4[j])
            ans.append(var3[j])

    return ans

