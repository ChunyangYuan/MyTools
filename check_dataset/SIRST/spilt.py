import cv2
import numpy as np
import mmcv
import mmengine
import os.path as osp
import os
import random


def split_dataset(dataset, split_ratio):
    random.shuffle(dataset)  # 随机打乱数据集
    split_index = int(len(dataset) * split_ratio)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]
    return train_set, test_set


def generate_txt_file(file_path, content_lines):
    with open(file_path, 'w') as file:
        for line in content_lines:
            file.write(line + '\n')


# misc数据集
input_folder = r'E:\dataset\SIRSTdevkit-master\Misc_ch3'
img_list = list(mmengine.scandir(input_folder))
dataset = [osp.splitext(img)[0] for img in img_list]
# img_list = os.listdir(input_folder)

# 划分比例
split_ratio = 0.8  # 80%用于训练集和验证，20%用于测试集

# 随机划分数据集
trainval_set, test_set = split_dataset(dataset, split_ratio)


# 设置文件路径和内容
trainval = 'misc3_trainval.txt'

test = 'misc3_test.txt'
# 生成txt文件
generate_txt_file(trainval, trainval_set)
generate_txt_file(test, test_set)
