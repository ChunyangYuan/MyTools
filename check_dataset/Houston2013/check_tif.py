import numpy as np
# import mmcv
import os
import os.path as osp
# import cv2
# import PIL.Image as Image
from skimage import io
# import spectral

# img_path = r'E:\dataset\2013_DFTC\2013_DFTC\2013_IEEE_GRSS_DF_Contest_LiDAR.tif'
img_path = r'E:\dataset\2013_DFTC\2013_DFTC\2013_IEEE_GRSS_DF_Contest_CASI.tif'
hdr = r'E:\dataset\2013_DFTC\2013_DFTC\2013_IEEE_GRSS_DF_Contest_CASI.hdr'
# 高光谱数据（多通道远大于4）mmcv, cv2, Image.open都无法读取
# img = mmcv.imread(img_path, flag='unchanged')
# img = cv2.imread(img_path)

# img = Image.open(img_path)
# data = np.array(img)


# 专门用于处理高光谱图像的库，例如spectral、scikit-image等


image = io.imread(img_path)
# img = spectral.open_image(hdr)
pass
