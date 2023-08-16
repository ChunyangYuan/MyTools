import scipy.io as sio
import numpy as np
import copy
import random


hsi_path = r'E:\dataset\HS-LiDAR data\Trento\HSI.mat'
hsi_mat = sio.loadmat(hsi_path)  # (166,600,63)
hsi_data = hsi_mat['HSI']

lidar_path = r'E:\dataset\HS-LiDAR data\Trento\LiDAR.mat'
lidar_mat = sio.loadmat(lidar_path)
lidar_data = lidar_mat['LiDAR']

train_label = r'E:\dataset\HS-LiDAR data\Trento\TRLabel.mat'
train_mat = sio.loadmat(train_label)
train_data = train_mat['TRLabel']
# for i in range(len(np.unique(train_data))):
#     print('No.{} train:{}'.format(i, np.sum(train_data == i)))
# print('='*50)
test_label = r'E:\dataset\HS-LiDAR data\Trento\TSLabel.mat'
test_mat = sio.loadmat(test_label)
test_data = test_mat['TSLabel']


# 查看样本数
num_samples = 0
for i in range(len(np.unique(test_data))):
    if i > 0:  # 不统计背景
        num_samples += np.sum(train_data == i)+np.sum(test_data == i)
    print('No.{} train:{} test:{} train+test:{}'.format(i,
          np.sum(train_data == i), np.sum(test_data == i), np.sum(train_data == i)+np.sum(test_data == i)))

print('='*50)
print("all samples:{}".format(num_samples))

# 合并样本数，再划分train和test
data_tr_va = copy.deepcopy(train_data)
# 将所有样本汇集到一起再划分数据集
for i in range(1, len(np.unique(train_data))):
    data_tr_va[test_data == i] = i

# 查看data_tr_va,可以发现
# for i in range(len(np.unique(train_data))):
#     print('No.{} train:{} val:{} train+val:{}'.format(i, np.sum(train_data == i),
#           np.sum(test_data == i), np.sum(train_data == i)+np.sum(test_data == i), np.sum(data_tr_va == i)))


# 划分训练集和测试集
data_tr_va = np.reshape(data_tr_va, -1)
data_train = copy.deepcopy(data_tr_va)
data_test = copy.deepcopy(data_tr_va)
num_train_samples = 20

for i in range(1, len(np.unique(data_tr_va))):
    idx_tr_va = np.where(data_tr_va == i)
    # print(type(idx_tr_va[0]))
    train_idx = random.sample(idx_tr_va[0].tolist(), 20)
    test_idx = list(set(idx_tr_va[0]) - set(train_idx))
    data_train[test_idx] = 0
    data_test[train_idx] = 0

# 恢复shape
data_train = np.reshape(data_train, train_data.shape)
data_test = np.reshape(data_test, train_data.shape)
# 查看数据
print('='*50)
for i in range(len(np.unique(train_data))):
    print('No.{} train:{}'.format(i, np.sum(data_train == i)))
print('='*50)
for i in range(len(np.unique(train_data))):
    print('No.{} test:{}'.format(i, np.sum(data_test == i)))


sio.savemat('train_test_gt.mat', {
            'train_data': data_train, 'test_data': data_test})
# sio.savemat('Trento_test.mat', {'test_data': data_test})
