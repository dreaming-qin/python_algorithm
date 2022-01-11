import matplotlib.pyplot as plt
import networkx as nx

# 打印有向图
# weight权重
# graph是一个二维列表，graph[i][j]代表的是i号节点指向j号节点
# index_dic是字典，key为i，value为num，它代表第i个节点的标签是num
def print_graph(weight,graph,index_dic):
    G = nx.generators.directed.random_k_out_graph(len(graph), 0, 0.5)
    pos = nx.layout.spring_layout(G)

    # 加边
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            G.add_edge(i,graph[i][j])

    labels={}
    for i in range(len(graph)):
        labels[i]=str(index_dic[i])

    nx.draw_networkx(G, pos, with_labels=None,node_size=weight)
    nx.draw_networkx_labels(G, pos, labels=index_dic)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()