import sys
from matplotlib import colors

import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import numpy as np
import random
from numpy.lib.function_base import delete
import umap
from scipy.special import comb

# k是聚类个数，data是n×m的数据矩阵，n是事务个数，m是事务特征个数
# 返回两个参数
# 第一个是一个二维数组，每一行代表的是同一类数据在data的下标
# 第二个是这个聚群的平均点
def k_means(data,k):
    data=np.array(data)
    # 对初始点的决定，随机选K个点做初始点
    avg_plot=[]
    for i in range(k):
        avg_plot.append(data[random.randint(0,len(data))])
    # 一直迭代，直到点的变化误差小于阈值为止
    # 误差阈值
    stop=0.000000001
    # 每个点所在的组
    group=[]
    while True:
        temp_group=[]
        for i in range(k):
             temp_group.append([])
        # 计算每个样本到平均点的距离
        for i in range(len(data)):
            min_distance=sys.maxsize
            min_index=-1
            for j in range(k):
                var1=np.sum(np.subtract(avg_plot[j],data[i])**2)
                if var1<min_distance:
                    min_distance=var1
                    min_index=j
            temp_group[min_index].append(i)
        # 根据每个点所属聚群再次计算新的平均点
        temp_avg_plot=[]
        for i in range(len(temp_group)):
            # 这个聚群中的所有维信息
            var2=[]
            for j in range(len(temp_group[i])):
                var2.append(data[temp_group[i][j]])
            temp_avg_plot.append(np.mean(var2,axis=0))
        # 查看新点是否满足误差条件，flag为TRUE时代表满足跳出循环
        flag=True
        for i in range(k):
            if np.sum(np.subtract(avg_plot[i],temp_avg_plot[i])**2)>stop:
                avg_plot=temp_avg_plot
                group=temp_group
                flag=False
                break
        if flag:
            break
    
    # 最后，将数组为空的地方去除
    row_index=[]
    for i in range(len(group)):
        if len(group[i])==0:
            row_index.append(i)
    group=np.delete(group,row_index,0)
    avg_plot=np.delete(avg_plot,row_index,0)

    return group,avg_plot


# 可视化，data是n×m的数据矩阵，n是事务个数，m是事务特征个数
# cluster_index是聚群列表，每一行代表的是在同一个集群在data的下标
# fig_name是保存的图片名称，为''时代表不把可视化的结果保存为图片
def visual(data,cluster_index,fig_name=''):
    # 对数据降维
    # reducer = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    reducer=umap.UMAP(random_state=42)
    var1=reducer.fit_transform(data)

    # 先打数据点
    # 产生k种不同的颜色
    colors = np.random.rand(len(cluster_index),4) # 随机产生k个0~1之间的颜色值
    # 点的大小
    area = np.pi * 2**2
    # 透明度
    alp=0.4
    for i in range(len(cluster_index)):
        if len(cluster_index[i])!=0:
            for j in range(len(cluster_index[i])):
                plt.scatter(var1[cluster_index[i][j]][0],var1[cluster_index[i][j]][1],s=area,c=colors[i],alpha=alp)
            plt.scatter(var1[cluster_index[i][j]][0],var1[cluster_index[i][j]][1],s=area,c=colors[i],alpha=alp,label='class'+str(i))
    # 然后打集群的中心点
    # 先获得集群中心点
    for i in range(len(cluster_index)):
        x=0
        y=0
        if len(cluster_index[i])!=0:
            for j in range(len(cluster_index[i])):
                x=x+var1[cluster_index[i][j]][0]
                y=y+var1[cluster_index[i][j]][1]
            x=x/len(cluster_index[i])
            y=y/len(cluster_index[i])
            plt.scatter(x,y,marker='+',c=colors[i],alpha=1,s=4*area)
    
    # 显示图像
    plt.legend()
    if fig_name!='':
        plt.savefig(fig_name)
    plt.close()

# 计算聚类的ARI指标
#返回ARI的值
def test(target,group):
    # 生成一个映射矩阵，将target映射到某一行中
    target_map={}
    cnt=0
    for i in range(len(target)):
        if target[i][0] not in target_map:
            target_map[target[i][0]]=cnt
            cnt=cnt+1

    # 先获得那个ARI矩阵
    # 先定义一个矩阵，并初始化它
    ARI=[]
    for i in range(cnt+1):
        ARI.append([])
        for j in range(len(group)+1):
            ARI[i].append(0)
    # 统计矩阵个数
    for i in range(len(group)):
        for j in range(len(group[i])):
            row=target_map[target[group[i][j]][0]]
            col=i
            ARI[row][col]=ARI[row][col]+1
            ARI[row][len(group)]=ARI[row][len(group)]+1
            ARI[cnt][col]=ARI[cnt][col]+1
            ARI[cnt][len(group)]=ARI[len(ARI)-1][len(ARI[i])-1]+1
    
    # 计算组合数
    for i in range(len(ARI)):
        for j in range(len(ARI[i])):
            ARI[i][j]=comb(ARI[i][j],2)
    
    # 计算四个需要的值
    var1=0
    for i in range(len(ARI)-1):
        for j in range(len(ARI[i])-1):
            var1=var1+ARI[i][j]
    var2=0
    for i in range(cnt):
        var2=var2+ARI[i][len(group)]
    var3=0
    for i in range(len(group)):
        var3=var3+ARI[cnt][i]
    var4=ARI[cnt][len(group)]
    ARI=(var1-(var2*var3)/var4)/(0.5*(var2+var3)-(var2*var3)/var4)

    return ARI
    
