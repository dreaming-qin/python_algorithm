import time
import os
from sklearn.decomposition import FastICA
import pandas as pd
import numpy as np
from k_means import k_means, test, visual
from write_to_csv import write_by_csv
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from scipy.special import comb

# 将文件转为csv文件
# str1='..'
# file_name_list = os.listdir(str1)
# for filename in file_name_list:
#     file_path=os.path.join(str1,filename)
#     if os.path.isfile(file_path) and 'rds' in file_path:
#         target=filename+'.csv'
#         target=os.path.join('csv_data',target)
#         write_by_csv(file_path,target)
#         print(filename+'转化完成')
# print('已全部转化为csv文件')
# print('---------------------------------------------------------------')


index=0
ks=[4,16,7,8,9]
str1='csv_data'
# ans存每个测试得到的精确度
ans={}

flag=True

start_time=time.time()

file_name_list = os.listdir(str1)
for filename in file_name_list:
    if 'label' not in filename:

        file_path=os.path.join(str1,filename)
        df=pd.read_csv(file_path,encoding='gbk')
        data=np.array(df.iloc[:,1:len(df.iloc[0])].values)

        # 对数据降维
        data=np.transpose(data)
        # ICA方法
        # reducer = FastICA(n_components=int(len(data)*0.8), random_state=42)
        # reduce_data=reducer.fit_transform(data)
        # UMAP方法
        # reducer=umap.UMAP(random_state=42,n_components=int(len(data)*0.8))
        # reduce_data=reducer.fit_transform(data)
        # PCA方法
        reducer=PCA(n_components=0.8)
        reduce_data=reducer.fit_transform(data)
        # 不用降维方法
        # reduce_data=data

        # 聚类
        group,avg_plot=k_means(reduce_data,ks[index])

        # 计算正确率
        # 获得target
        str2=list(file_path)
        str2.insert(len(str2)-4,'_label')
        str2=''.join(str2)
        df=pd.read_csv(str2,encoding='gbk')
        target=np.array(df.iloc[:,1:len(df.iloc[0])].values)
        # 计算正确率的函数
        ARI=test(target,group)
        # 存到ans中
        ans[filename]=ARI

        # 可视化
        visual(reduce_data,group,'cluster'+str(index)+'.png')
        print(str(index)+' is finished')
        index=index+1

end_time=time.time()

print('--------------------------------------------------------------')
print('running time is '+str((end_time-start_time)/60)+' minutes')
print('--------------------------------------------------------------')
for key in ans:
    print("the ARI of "+key+' is '+str(ans[key]))
