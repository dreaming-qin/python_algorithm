# 通过数据data返回决策树模型
from math import log2
from graphviz import Digraph
import numpy as np
from numpy.core.fromnumeric import var
from numpy.lib.function_base import gradient
import csv
import pandas as pd
import os


root = "0"

# 实现的是C4.5算法
class desicion_tree:
    # 树模型，封装到类中
    tree_modle=0

    # 外部调用
    def fit(self,attr,target):
        attr_name=[]
        for i in range(0,len(attr[0])):
            attr_name.append("a"+str(i))
        attr_name=np.array(attr_name)
        var1 = self.getTreeList(attr, attr_name, target)
        root = {"root": var1}
        self.tree_modle=root

    # 真正实现，因为是树，所以递归调用
    def getTreeList(self, attr, attr_name,target):
        # 验证这个数据纯不纯
        # 首先，看attr纯不纯
        if len(attr[0])==0:
            return "class="+str(np.argmax(np.bincount(target)))
        flag=False
        for i in range(0, len(attr)):
            for j in range(0,len(attr[0])):
                if i == len(attr)-1:
                    return "class="+str(np.argmax(np.bincount(target)))
                if attr[i][j] != attr[i+1][j]:
                    flag=True
                    break
            if flag:
                break
        # 其次，看class纯不纯
        for i in range(0, len(target)):
            if i == len(target)-1:
                return "class="+str(target[0])
            if target[i] != target[i+1]:
                break

        # 第一步，计算每个元素的熵值
        # dic是一个熵值数组，dic[i]代表的是对应第i个属性的字典，字典的key是属性值，value是对应的数量
        # value也是一个字典，key是target的值，value是对应数量
        # 有点套
        dic = []
        for i in range(0, len(attr[0])):
            dic.append(dict())
        
        # 开始获取熵值和属性自身的内部信息增益
        # 首先，获取每个属性的对应类的数量，存在dic中
        for i in range(0, len(attr)):
            for j in range(0, len(attr[0])):
                if attr[i][j] in dic[j]:
                    var1 = dic[j][attr[i][j]]
                    if target[i] in var1:
                        var1[target[i]] = var1[target[i]]+1
                    else:
                        var1[target[i]] = 1
                else:
                    var1 = dic[j]
                    var1[attr[i][j]] = dict()
                    var1[attr[i][j]][target[i]] = 1
        # 然后，计算各个属性的熵值，存在info中
        info = []
        # H是内部的信息增益
        H = []
        for i in range(0, len(dic)):
            info.append(0)
            H.append(0)
            for key in dic[i]:
                cnt = 0
                temp_ans = 0
                var1 = dic[i][key]
                for key2 in var1:
                    cnt = cnt+var1[key2]
                for key2 in var1:
                    temp_ans = temp_ans-(var1[key2]/cnt)*log2(var1[key2]/cnt)
                info[i] = info[i]+(cnt/len(attr))*temp_ans
                H[i] = H[i]-(cnt/len(attr))*log2(cnt/len(attr))
        # GAIN是这个数据的总信息熵
        GAIN=0
        var2={}
        for key in dic[0]:
            var1= dic[0][key]
            for key2 in var1:
                if key2 in var2:
                    var2[key2]=var2[key2]+var1[key2]
                else:
                    var2[key2]=var1[key2]
        for key in var2:
            GAIN=GAIN-(var2[key]/len(attr))*log2(var2[key]/len(attr))
        GAIN=np.subtract(GAIN,info)
        IGR = np.divide(GAIN, np.add(H,1.1920929e-07))
        # 经过上述步骤，熵值和属性内部信息增益算完，获得信息增益率存在IGR中

        # 获得最小的熵值所在的属性下标
        IGR=IGR.tolist()
        max_index = IGR.index(max(IGR))

        # 以这个下标进行分割，然后递归计算，获得树，存到ans中
        ans = {}
        # 让数据根据这个元素进行排列
        data = attr.tolist()
        for i in range(0,len(attr)):
            data[i].append(target[i])
        data.sort(key=lambda data: data[max_index], reverse=False)
        # 减少这一列的数据，为下一次做准备
        var3=data
        data=np.delete(data,max_index,1)
        data = np.array(data)
        max_attr_name=attr_name[max_index]
        attr_name=np.delete(attr_name,max_index,0)

        # 开始递归计算，获得子树
        var5={}
        s_index = 0
        for i in range(0, len(data)):
            if i == len(data)-1 or var3[i][max_index] != var3[i+1][max_index]:
                var1 = str(max_attr_name)+"="+str(var3[i][max_index])
                var2 = self.getTreeList(data[s_index:i+1, 0:len(data[0])-1], attr_name,data[s_index:i+1, len(data[0])-1])
                if isinstance(var2,str):
                    ans.update({var1: var2})
                else:
                    var4={}
                    var4.update({'':var2})
                    ans.update({var1: var4})
                s_index = i+1
        
        flag=True
        # 对ans进行查重
        for key in ans:
            var1=ans[key]
            break
        for key in ans:
            if var1!=ans[key] or (not isinstance(var1,str)):
                flag=False
                break
        if flag:
            ans=var1
        
        # 返回ans
        return ans

    # 下面用于树的可视化
    def print_tree_to_file(self, file_name):
        tree=self.tree_modle
        g = Digraph("G", filename=file_name, format='png', strict=False)
        first_label = list(tree.keys())[0]
        g.node("0", first_label)
        self._sub_plot(g, tree, "0")
        g.view()
        # 删除临时文件
        os.remove('tree')

    # 递归打点
    def _sub_plot(self,g,tree, inc):
        global root
        first_label = list(tree.keys())[0]
        ts = tree[first_label]
        for i in ts.keys():
            if isinstance(tree[first_label][i], dict):
                root = str(int(root) + 1)
                g.node(root, list(tree[first_label][i].keys())[0])
                g.edge(inc, root, str(i))
                self._sub_plot(g, tree[first_label][i], root)
            else:
                root = str(int(root) + 1)
                g.node(root, tree[first_label][i])
                g.edge(inc, root, str(i))

    # 用测试集进行测试，返回的是正确率
    def test(self,attr_test,target_test):
        # 获得属性名和坐标的映射
        attr_map={}
        for i in range(0,len(attr_test[0])):
            attr_map.update({'a'+str(i):i})
        # ans记录预测正确的个数
        ans=0
        # 遍历树，找类
        for i in range(len(attr_test)):
            var1=self.tree_modle['root']
            while True:
                for key in var1:
                    str1=key
                    break
                str1=str1[0:str1.index('=')]
                str1=str1+'='+str(attr_test[i][attr_map[str1]])
                # 防止树中没有这个属性，于是跳出
                if str1 not in var1:
                    var1='='
                    break
                var1=var1[str1]
                if isinstance(var1,str):
                    break
                else:
                    var1=var1['']
            var1=var1[var1.index('=')+1:]
            # 如果与实际情况是一样的，ans加1
            if var1==str(target_test[i]):
                ans=ans+1
        
        # 返回正确率
        return ans/len(target_test)
            
