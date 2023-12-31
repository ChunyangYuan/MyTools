import scipy.io as sio
import PIL.Image as Image
import numpy as np
import os.path as osp
from skimage import io


def generate_Houston_mat(tif_path: str, output_folder: str):
    img = io.imread(tif_path)  # img_data.shape=(349,1905,144)
    # img_data = np.array(img)
    save_path = osp.join(output_folder, 'Houston.mat')
    sio.savemat(save_path, {'img': img})

    print('Houston.mat is generated!')
    pass


def generate_LiDAR_mat(tif_path: str, output_folder: str):
    img = Image.open(tif_path)
    img_data = np.array(img)  # img_data.shape=(349,1905)
    save_path = osp.join(output_folder, 'LiDAR.mat')
    sio.savemat(save_path, {'img': img_data})
    print('LiDAR.mat is generated!')
    pass


def generate_train_test_gt_1_mat(train_tif: str, test_tif: str, output_folder: str):

    train_img = Image.open(train_tif)
    val_img = Image.open(test_tif)

    train_data = np.array(train_img)
    val_data = np.array(val_img)

    save_path = osp.join(output_folder, 'train_test_gt_1.mat')
    sio.savemat(save_path, {'train_data': train_data, 'test_data': val_data})
    print('train_test_gt.mat is generated!')
    pass


lidar_tif = r'E:\dataset\HS-LiDAR data\Houston2013\2013_IEEE_GRSS_DF_Contest_LiDAR.tif'
# CASI数据是由Compact Airborne Spectrographic Imager（紧凑型机载光谱成像仪）获取的
casi_tif = r'E:\dataset\HS-LiDAR data\Houston2013\2013_IEEE_GRSS_DF_Contest_CASI.tif'
train_tif = r'F:\PythonProjects\MyTools\Houston2013_train.tif'
test_tif = r'F:\PythonProjects\MyTools\Houston2013_test.tif'
save_dir = r'E:\dataset\HS-LiDAR data\Houston2013'

# generate_Houston_mat(casi_tif, save_dir)
# generate_LiDAR_mat(lidar_tif, save_dir)
generate_train_test_gt_1_mat(train_tif, test_tif, save_dir)
