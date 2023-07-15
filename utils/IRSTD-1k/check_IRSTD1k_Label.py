import numpy as np
# import mmcv
import os
import os.path as osp
# import cv2
import PIL.Image as Image


def check_label(input_folder: str):
    img_list = os.listdir(input_folder)
    for i in range(len(img_list)):
        img_list[i] = osp.join(input_folder, img_list[i])

    for img_path in img_list:
        img = Image.open(img_path)
        img = np.array(img)
        unique_list = np.unique(img)
        print(unique_list)
        pass
    pass


img_dir = r'E:\dataset\IRSTD-1k\IRSTD1k_Label'
check_label(img_dir)
pass
