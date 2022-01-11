import numpy as np
# graph是一个二维列表，graph[i][j]=k代表的是i号节点指向k号节点
# weight是权重，是一个二维列表weight[i][j]代表边ij的权重
# d是阻尼系数
def page_rank(graph,weight=None,d=0.9):
    if weight is None:
        weight=[]
        var1=1/len(graph)
        for i in range(len(graph)):
            weight.append(var1)
    
    # 计算矩阵M
    M = [[0] * len(graph) for row in range(len(graph))]
    for i in range(len(graph)):
        var1=len(graph[i])
        if var1!=0:
            for j in range(len(graph[i])):
                M[graph[i][j]][i]=1
    
    init_weight=np.dot(M,np.array(weight))
    R=weight
    stop=0.0000000001
    # 开始迭代
    while True:
        var1=np.dot(M,R)
        var1=var1/np.sum(var1)
        var1=d*var1+(1-d)*init_weight
        var1=var1/np.sum(var1)
        var2=np.subtract(var1,R)
        if(np.sum(np.maximum(var2,-var2))<stop):
            break
        R=var1

    # 归一化
    R=R/np.sum(R)
    return R