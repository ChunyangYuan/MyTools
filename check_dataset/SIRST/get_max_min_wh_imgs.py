import cv2
import numpy as np
import mmcv
import mmengine
import os.path as osp
# from typing import tuple


def get_max_min_wh_in_img_dir(input_folder: str):
    w_max = 0
    w_min = 10000
    h_max = 0
    h_min = 10000
    img_list = list(mmengine.scandir(input_folder))
    img_list = [osp.join(input_folder, v) for v in img_list]
    cnt1 = 0
    cnt3 = 0
    cnt33 = 0
    for img in img_list:
        img = mmcv.imread(img, flag='unchanged')  # h,w
        # img2 = mmcv.imread(img)  # h,w,3
        if len(img.shape) == 2:
            cnt1 += 1
            height, width = img.shape
        elif len(img.shape) == 3:
            cnt3 += 1
            if img.shape[-1] == 3:
                cnt33 += 1
            height, width, _ = img.shape
        else:
            print(
                f'len(img.shape) = {img.shape}, only support len(img.shape)==2 or 3!')
            sys.exit(1)
        if height > h_max:
            h_max = height
        elif height < h_min:
            h_min = height

        if width > w_max:
            w_max = width
        elif width < w_min:
            w_min = width

    print("h_min={},h_max={},w_min={},w_max={}".format(
        h_min, h_max, w_min, w_max))
    print(cnt1, cnt3, cnt33)
    # return (h_min, h_max, w_min, w_max)


if __name__ == "__main__":
    img_dir = r'F:\dataset\SIRSTdevkit-master\Misc'
    get_max_min_wh_in_img_dir(img_dir)
    pass
