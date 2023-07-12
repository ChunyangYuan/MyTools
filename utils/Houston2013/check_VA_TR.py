import numpy as np
# import mmcv
import os
import os.path as osp
# import cv2
import PIL.Image as Image

img_va = r'E:\dataset\HS-LiDAR data\Houston2013\2013_IEEE_GRSS_DF_Contest_Samples_VA.tif'
img_tr = r'E:\dataset\HS-LiDAR data\Houston2013\2013_IEEE_GRSS_DF_Contest_Samples_TR.tif'
# img = mmcv.imread(img_path, flag=0)  # h,w
# img = cv2.imread(img_path)  # h,w
img_va = Image.open(img_va)
img_tr = Image.open(img_tr)
data_va = np.array(img_va)
data_tr = np.array(img_tr)
print(data_va.size)
print(np.sum(data_va == data_tr))
pass
