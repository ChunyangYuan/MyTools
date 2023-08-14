import numpy as np
# import mmcv
import os
import os.path as osp
# import cv2
import PIL.Image as Image
import copy
import random


img_va = r'E:\dataset\HS-LiDAR data\Houston2013\2013_IEEE_GRSS_DF_Contest_Samples_VA.tif'
img_tr = r'E:\dataset\HS-LiDAR data\Houston2013\2013_IEEE_GRSS_DF_Contest_Samples_TR.tif'

img_va = Image.open(img_va)
img_tr = Image.open(img_tr)
data_va = np.array(img_va)
data_tr = np.array(img_tr)
# print(data_tr.shape)
# 查看所有标签样本
# No.表示类别标签,train表示训练样本,val表示验证样本,train+val表示训练+验证样本.
# 查看输出可以发现train+val的样本数与论文表格1中的总样本数一致！
for i in range(len(np.unique(data_tr))):
    print('No.{} train:{} val:{} train+val:{}'.format(i,np.sum(data_tr == i), np.sum(data_va==i),np.sum(data_tr == i)+np.sum(data_va==i)))

data_tr_va = copy.deepcopy(data_tr)
print(np.sum(data_va == data_tr))
# 将所有样本汇集到一起再划分数据集
print("="*50)
for i in range(1,len(np.unique(data_tr))):
    data_tr_va[data_va == i] = i

# 查看data_tr_va,可以发现
for i in range(len(np.unique(data_tr))):
    print('No.{} train:{} val:{} train+val:{}'.format(i,np.sum(data_tr == i), np.sum(data_va==i),np.sum(data_tr == i)+np.sum(data_va==i), np.sum(data_tr_va == i)))
img = Image.fromarray(data_tr_va)
# img.save('Houston2013_data_tr_va.tif')

# 划分训练集和测试集
data_tr_va = np.reshape(data_tr_va,-1)
data_train = copy.deepcopy(data_tr_va)
data_test = copy.deepcopy(data_tr_va)
num_train_samples = 20

for i in range(1,len(np.unique(data_tr_va))):
    idx_tr_va = np.where(data_tr_va == i)
    print(type(idx_tr_va[0]))
    train_idx = random.sample(idx_tr_va[0].tolist(),20)
    test_idx = list(set(idx_tr_va[0]) - set(train_idx))
    data_train[test_idx] = 0
    data_test[train_idx] = 0
    
# 恢复shape
data_train = np.reshape(data_train, data_tr.shape)
data_test = np.reshape(data_test, data_tr.shape)
# 查看数据
for i in range(len(np.unique(data_tr))):
    print('No.{} train:{}'.format(i, np.sum(data_train == i)))
for i in range(len(np.unique(data_tr))):
    print('No.{} test:{}'.format(i,np.sum(data_test == i)))


train_img = Image.fromarray(data_train)
test_img = Image.fromarray(data_test)
train_img.save('Houston2013_train.tif')
test_img.save('Houston2013_test.tif')
