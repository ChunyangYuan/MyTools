import cv2
import numpy as np
import mmcv
import mmengine
import os.path as osp
import os


def pick_3channel_img(input_folder: str, output_folder: str):
    if not osp.exists(input_folder):
        print('input_folder do not exist! please check out!')
        sys.exit(1)

    if not osp.exists(output_folder):
        os.mkdir(output_folder)

    img_list = list(mmengine.scandir(input_folder))
    # img_list = os.listdir(input_folder)
    img_list = [osp.join(input_folder, v) for v in img_list]

    for img_path in img_list:
        img = mmcv.imread(img_path, flag='unchanged')
        if len(img.shape) == 3 and img.shape[-1] == 3:
            # if len(img.shape) == 2:
            mmcv.imwrite(img, osp.join(output_folder, osp.basename(img_path)))
    else:
        print(r'pick over!')


data_path = r'F:\dataset\SIRSTdevkit-master\Misc'
output = r'F:\dataset\SIRSTdevkit-master\Misc_ch3'
pick_3channel_img(data_path, output)
