import numpy as np
# import mmcv
import os
import os.path as osp
# import cv2
import PIL.Image as Image

img_path = r'E:\dataset\HS-LiDAR data\Houston2013\2013_IEEE_GRSS_DF_Contest_Samples_TR.tif'
# img = mmcv.imread(img_path, flag=0)  # h,w
# img = cv2.imread(img_path)  # h,w
img = Image.open(img_path)
data = np.array(img)
pass
