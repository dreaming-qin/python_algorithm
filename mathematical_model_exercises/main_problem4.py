import pandas as pd
import numpy as np
import math
import sys
import random

from term_influnce import format_data, get_follower, get_music_feature, get_term_MS

filename='../2021_ICM_Problem_D_Data/full_music_data.csv'
df = pd.read_csv(filename)
data = np.array(df.iloc[:,:].values)

# 名称
influnce_name='The Rolling Stones'
follower_name_set=set(['Devo','Alice Cooper'])
# follower_name_set=set(['The Stone Roses','Alice Cooper'])
# 不正确的样例，可以用作对比
influnce_name='Black Box'
follower_name_set=set(['David Guetta','Soda Stereo'])

# 第四题第一部分
str1='influncer,follower,LTS,STS\n'
for follower_name in follower_name_set:
    MS=get_term_MS(data,influnce_name,follower_name,step=3)
    delta_MS=[MS[i+1]-MS[i] for i in range(len(MS)-1)]
    STS=np.max(delta_MS)
    LTS=np.sum(delta_MS)
    str1=str1+"{},{},{},{}\n".format(influnce_name,follower_name,LTS,STS)

print(str1)
file=open('data/problem4_term_influnce.csv','w',errors='ignore')
file.write(str1)
file.close()

# 第四题第二部分
str1=''
for follower_name in follower_name_set:
    attr_name,score=get_music_feature(data,influnce_name,follower_name)
    str1=str1+"{},{}".format(influnce_name,follower_name)
    for item in score:
        str1=str1+',{}'.format(item)
    str1=str1+'\n'
str1='\n'+str1
for i in range(len(attr_name)-1,-1,-1):
    str1=',{}'.format(attr_name[i])+str1
str1='influnce_name,follower_name'+str1


print(str1)
file=open('data/problem4_music_feature.csv','w',errors='ignore')
file.write(str1)
file.close()
