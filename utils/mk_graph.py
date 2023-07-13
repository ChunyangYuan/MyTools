import scipy.io as sio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import scipy.sparse as ss
import pandas as pd


def same_color(hq: str):
    hq = sio.loadmat(hq)
    data = hq['association_matrix'][0]
    mapping = []
    for i in range(1, len(data)):
        # print(data[i])
        row, col = np.where(data[i])
        for i in range(len(col)):
            print('{}'.format(col[i]))
        print("="*30)
        mapping.append(col)

    return mapping


def mk_graph(mat: str, mapping):
    mat_data = sio.loadmat(mat)
    colormaps = np.random.rand(100, 3)
    datas = mat_data['adj'][0]

    for i in range(len(datas)):
        data = np.array(datas[i], dtype=np.int64)
        if i < len(datas)-1:
            colors = random.sample(list(colormaps), len(data)//2)
        else:
            colors = random.sample(list(colormaps), len(data))
        if i < len(mapping):
            colors = np.array(colors)[mapping[i]]
        else:
            colors = np.array(colors)[:6, :]
        graph = nx.from_numpy_array(data)
        nx.draw(graph, node_color=colors)
        # plt.show()
        plt.savefig(str(len(data))+'.png', dpi=500, facecolor=(0, 0, 0, 0))
        plt.cla()
    else:
        print('mk graph over!')


def get_sparse_adj(mat: str):
    mat = sio.loadmat(mat)
    mat_data = mat['adj'][0]
    coo_mtx = []
    for i in range(len(mat_data)):
        coo_mtx.append(ss.coo_matrix(np.array(mat_data[i], dtype=np.int64)))
    return coo_mtx


if __name__ == "__main__":
    mat = r'C:\Users\LwhYcy\Desktop\U\segments\mat_pro\hierarchy_adjacency.mat'
    hq = r'C:\Users\LwhYcy\Desktop\U\segments\mat_pro\hierarchy_association_matrix.mat'

    pass
    # mat2
    # group = [[0, 1, 2, 3, 4, 1, 3, 3, 5, 6, 7, 1, 2, 8, 2, 8, 9, 9, 10, 11, 12, 13, 14, 11, 15,
    #          16, 17, 18, 15, 14, 15, 15, 14, 15, 15, 19, 20, 21, 19, 22, 21, 15, 23, 15, 19, 19, 23, 19],
    #          [0, 0, 1, 2, 3, 4, 5, 1, 2, 6, 7, 8, 5,
    #              8, 9, 9, 5, 7, 8, 10, 5, 11, 5, 11],
    #          [0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5],
    #          [0, 1, 2, 3, 4, 5]]

#   nodes: [
#     {id: "Richard", group: "family"},
#     {id: "Larry",   group: "family"},
#     {id: "Marta",   group: "family"},
#     {id: "Jane",    group: "friends"},
#     {id: "Norma",   group: "friends"},
#     {id: "Frank",   group: "friends"},
#     {id: "Brett",   group: "friends"},
#     {id: "Tommy",   group: "lone wolf"}
#   ],
#   edges: [
#     {from: "Richard", to: "Larry"},
#     {from: "Richard", to: "Marta"},
#     {from: "Larry",   to: "Marta"},
#     {from: "Marta",   to: "Jane"},
#     {from: "Jane",    to: "Norma"},
#     {from: "Jane",    to: "Frank"},
#     {from: "Jane",    to: "Brett"},
#     {from: "Brett",   to: "Frank"}
#   ]
    # coo_mtx = get_sparse_adj(mat)
    # source_list = []
    # target_list = []

    # data = {
    #     "Source point": source_list,
    #     "Target point": target_list
    # }

    # with pd.ExcelWriter(r'C:\Users\LwhYcy\Desktop\U\segments\mat_pro\segments-pro.xlsx', mode='w', engine='openpyxl') as writer:
    #     for n in range(4):
    #         source_list.clear()
    #         target_list.clear()
    #         #   nodes
    #         # print('=============  {}  nodes'.format(len(group[n])))
    #         # for i in range(len(group[n])):
    #         #     print('{{id: "{}", group: "g{}"}},'.format(i, group[n][i]))

    #         # edges
    #         print('=============  {}  edges'.format(coo_mtx[n].nnz))
    #         for i in range(coo_mtx[n].nnz):
    #             # print('{{from: "{}", to: "{}"}},'.format(
    #             #     coo_mtx[n].row[i], coo_mtx[n].col[i]))
    #             source_list.append(coo_mtx[n].row[i])
    #             target_list.append(coo_mtx[n].col[i])
    #         df = pd.DataFrame(data)

    #         df.to_excel(writer, sheet_name='sheet'+str(n+1))

# var family = chart.group("family");
    # for i in range(24):
    # print('var group_{} = chart.group("g{}");'.format(i, i))
    # print('group_{}.normal().fill("#ffa000");'.format(i))
    #  group_0.normal().fill("#ffa000")
    # print('group_{}.normal().height(20);'.format(i))
    # family.normal().height(40);
    # mapping = same_color(hq)
    # mk_graph(mat, mapping)

    mapping = same_color(hq)
    grp1 = mapping[0]
    grp2 = mapping[1]
    grp3 = mapping[2]
    grp4 = [0, 1, 2, 3, 4, 5, 6]
    group = {
        'group1': grp1,
        'group2': grp2,
        'group3': grp3,
        'group4': grp4
    }

    # with pd.ExcelWriter(r'C:\Users\LwhYcy\Desktop\U\segments\mat_pro\segments-pro.xlsx', mode='a', engine='openpyxl') as writer:
    #     df = pd.DataFrame(group)
    #     df.to_excel(writer, sheet_name='group')
