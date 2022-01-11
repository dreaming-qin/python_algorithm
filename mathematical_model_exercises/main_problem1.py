import pandas as pd
import numpy as np
import math
import random
from page_rank import page_rank
from print_graph import print_graph


filename='../2021_ICM_Problem_D_Data/influence_data.csv'
df = pd.read_csv(filename)
data = np.array(df.iloc[:,:].values)
# 先建图
num_dic={}
index_dic={}
index_to_name_dic={}
graph=[]
var1=0
for i in range(len(data)):
    if data[i][4] not in num_dic:
        index_to_name_dic[var1]=data[i][5].replace(',',' and ')
        num_dic[data[i][4]]=var1
        index_dic[var1]=data[i][4]
        var1=var1+1
        graph.append([])
    if data[i][0] not in num_dic:
        index_to_name_dic[var1]=data[i][1].replace(',',' and')
        num_dic[data[i][0]]=var1
        index_dic[var1]=data[i][0]
        var1=var1+1
        graph.append([])
    graph[num_dic[data[i][4]]].append(num_dic[data[i][0]])
# 再获得权重
weight = [[0] * len(graph) for row in range(len(graph))]

for i in range(len(data)):
    var1=int(data[i][7]-data[i][3])
    if var1<0:
        var1=-1.5*var1
    var2=0
    if data[i][2]==data[i][6]:
        var2=1
    weight[num_dic[data[i][0]]][num_dic[data[i][4]]]=(1+var2)*math.exp(var1/80)
var2=[0]*len(graph)
for i in range(len(weight)):
    var2[i]=np.sum(weight[i])
# 归一化
weight=var2/np.sum(var2)

# # 获得每个结点的权值
weight=page_rank(graph,weight,0.85)
weight=weight*100
# 保存到csv
# var1=[index_dic[i] for i in range(len(weight))]
# des_file=open('data/influence_artist.csv','w',errors='ignore')
# str1='id,name,secore\n'
# for i in range(len(weight)):
#     str1=str1+str(var1[i])+','+index_to_name_dic[i]+','+str(weight[i])+'\n '
#     if i%1000==0:
#         des_file.write(str1)
#         str1=''
# des_file.write(str1)
# des_file.close()

# 画图
# print_graph(weight,graph,index_to_name_dic)

# 状态码
# 登录逻辑
# 页面传参
# 组件
# 页面返回

src_set=[]
for i in range(len(graph)):
    for j in range(len(graph[i])):
        if index_dic[graph[i][j]]==754032:
            src_set.append([index_dic[i],weight[i]])
            break
src_set.append([754032,1.978474059])
src_set.sort(key=lambda src_set: src_set[1], reverse=True)
var1=set()
for i in range(30):
    var1.add(src_set[i][0])
src_set=var1


# # 存plot.csv
# str1='Id,Label,weight,Modul arity class\n'
# for i in range(len(graph)):
#     if index_dic[i] in src_set:
#         str1='{},{},{}\n'.format(index_dic[i],index_to_name_dic[i],weight[i])
# file=open('data/plot.csv','w',errors='ignore')
# file.write(str1)
# file.close()

file=open('data/plot.csv','w',errors='ignore')
# 存plot.csv
str1='Id,Label,weight\n'
for i in range(len(graph)):
    if index_dic[i] in src_set:
        str1+='{},{},{:.2f}\n'.format(index_dic[i],index_to_name_dic[i],weight[i])
        if i%1000==0:
            file.write(str1)
            str1=''
file.write(str1)
file.close()

file=open('data/edge.csv','w',errors='ignore')
str1='Source,Target,Type,Id\n'
var1=0
for i in range(len(graph)):
    for j in range(len(graph[i])):
        if index_dic[i] in src_set and index_dic[graph[i][j]] in src_set:
            str1+='{},{},Directed,{}\n'.format(index_dic[i],index_dic[graph[i][j]],var1)
            var1+=1
            if var1%1000==0:
                file.write(str1)
                str1=''
file.write(str1)
file.close()

# print(ans)
# file=open('data/problem3.csv','w',errors='ignore')
# file.write(ans)
# file.close()
# a=1
    





