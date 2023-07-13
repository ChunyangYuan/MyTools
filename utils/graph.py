import networkx as nx
import matplotlib.pyplot as plt
import random

# 创建一个空图
G = nx.Graph()

# 添加30个节点
num_nodes = 30
G.add_nodes_from(range(num_nodes))

# 添加至少两条边到每个节点
for node in G.nodes():
    # 确保每个节点至少有两条边
    while G.degree(node) < 2:
        # 随机选择另一个节点并添加一条边
        random_node = node
        while random_node == node or G.has_edge(node, random_node):
            random_node = random.choice(list(G.nodes()))
        G.add_edge(node, random_node)

# 绘制图形
pos = nx.spring_layout(G)  # 选择布局算法
nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_weight='bold')
plt.title("Graph Visualization")
plt.show()
