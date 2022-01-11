import pandas as pd
import numpy as np

from MS import global_

def get_term_MS(data,influnce_name,follower_name,window_size=5,step=5):
    influnce_data,follower_data=get_data_byname(data,influnce_name),get_data_byname(data,follower_name)
    var1=[0,1,7,8,13,16,18]
    influnce_data,follower_data=np.delete(influnce_data,var1,1),np.delete(follower_data,var1,1)
    # 对日期格式化以便排序
    influnce_data,follower_data=format_date(influnce_data),format_date(follower_data)
    influnce_data,follower_data=influnce_data.tolist(),follower_data.tolist()

    var1=len(influnce_data[0])-1
    influnce_data.sort(key=lambda influnce_data: influnce_data[var1], reverse=False)
    follower_data.sort(key=lambda follower_data: follower_data[var1], reverse=False)
    # 对数据进行归一化
    # influnce_data,influnce_data_date=format_data_date(influnce_data)
    # follower_data,follower_data_date=format_data_date(follower_data)
    var1,var2=np.transpose(influnce_data),np.transpose(follower_data)
    influnce_data,follower_data=np.delete(influnce_data,len(influnce_data[0])-1,1),np.delete(follower_data,len(follower_data[0])-1,1)
    influnce_data,influnce_data_date=influnce_data[0:len(influnce_data),:],var1[len(var1)-1]
    follower_data,follower_data_date=follower_data[0:len(follower_data)-1,:],var2[len(var2)-1]
    # 开始构建
    var1=int((len(follower_data)-window_size)/step)+1
    MS=[0 for i in range(var1)]
    start=0
    end=start+window_size-1
    # 使用指数平滑算法，设定alpha
    alpha=0.2
    for i in range(len(MS)):
        var_data=follower_data[start:end+1,:]
        var_data=var_data.mean(axis=0)
        influnce_data_index=0
        # 进行指针运算
        while True:
            if influnce_data_index==len(influnce_data)-1 or influnce_data_date[influnce_data_index]>=follower_data_date[start]:
                break
            influnce_data_index=influnce_data_index+1
        if influnce_data_date[influnce_data_index]>=follower_data_date[start]:
            influnce_data_index=influnce_data_index-1
        # 让influnce数据小于这首歌之前的所有歌曲进行权值相加
        var1=alpha
        score=0
        for j in range(influnce_data_index,-1,-1):
            score=score+var1*global_(var_data,influnce_data[j])
            var1=var1*(1-alpha)
        MS[i]=score
        start,end=start+step,end+step
    
    var1=[]
    for i in range(len(MS)):
        if MS[i]!=0:
            break
        var1.append(i)

    return np.delete(MS,var1,0)

# 返回名字和对应值
def get_music_feature(data,influnce_name,follower_name):
    filename='../2021_ICM_Problem_D_Data/full_music_data.csv'
    df = pd.read_csv(filename)
    attr=np.array(df.columns).tolist()

    influnce_data=get_data_byname(data,influnce_name)
    follower_data=get_data_byname(data,follower_name)

    var1=[0,1,7,8,13,16,17,18]
    influnce_data=np.delete(influnce_data,var1,1)
    follower_data=np.delete(follower_data,var1,1)
    attr=np.delete(attr,var1,0)
    # 对数据进行归一化
    # influnce_data=format_data_feature(influnce_data)
    # follower_data=format_data_feature(follower_data)

    score=[0 for i in range(len(attr))]
    influnce_data=np.transpose(influnce_data)
    follower_data=np.transpose(follower_data)
    # 以一个特征为目标
    dif=[0 for jkj in range(len(follower_data[0]))]
    for i in range(len(attr)):
        for j in range(len(follower_data[i])):
            var2=follower_data[i][j]-influnce_data[i]
            var1=np.abs(var2)
            var1=var1.tolist()
            var1.sort(reverse=False)
            dif[j]=0.5*var1[0]+0.3*var1[1]+0.2*var1[2]
        avg=np.mean(dif,axis=0)
        variance=np.std(dif)
        var1=np.min(dif)
        score[i]=((avg*1000000)*(variance*1000000)*(var1*1000000))/10
    return attr,score

def get_data_byname(data,music_name):
    data_relation=[]
    for item in data:
        if music_name in item[0]:
            data_relation.append(item)
    return data_relation

def format_date(data):
    for i in range(len(data)):
        if data[i][len(data[i])-1].count('/')!=2:
            data[i][len(data[i])-1]='00/00/'+data[i][len(data[i])-1]
    for item in data:
        str1=item[len(item)-1]
        str1=str1.split('/')
        if len(str1[0])==1:
            str1[0]='0'+str1[0]
        if len(str1[1])==1:
            str1[1]='0'+str1[1]
        str1=str1[2]+str1[0]+str1[1]
        item[len(item)-1]=int(str1)
    return data

# # 归一化数据
# def format_data_date(data):
#     data=np.transpose(data)
#     var1=[data[i]/np.sum(data[i]) for i in range(len(data)-1)]
#     var1=np.transpose(np.array(var1))
#     return var1,data[len(data)-1]

# # 归一化数据
# def format_data_feature(data):
#     data=np.transpose(data)
#     var1=[data[i]/np.sum(data[i]) for i in range(len(data))]
#     data=np.transpose(np.array(var1))
#     data=[data[i]/np.sum(data[i]) for i in range(len(data))]
#     return data

def format_data(data,index_list,row):
    if row:
        if isinstance(data,np.ndarray):
            data=data.tolist()
        for index in index_list:
            data[index]=data[index]/np.sum(data[index])
    else:
        var1=data
        var1=np.transpose(var1)
        for index in index_list:
            data[:,index]=var1[index]/np.sum(var1[index])
    return data

def get_follower(influncer_name):
    filename='../2021_ICM_Problem_D_Data/influence_data.csv'
    df = pd.read_csv(filename)
    data = np.array(df.iloc[:,:].values)
    follower=[]
    for item in data:
        if influncer_name in item[1]:
            follower.append(item[5])
    return follower