import numpy as np
from PIL import Image
import cv2
import os


def split_img_with_overlap(img_path:str, save_dir:str, width:int, num_w:int, overlap_width:int=0,  num_h:int=None,height:int=None):
    """
    从左往右, 从上往下切图
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img = Image.open(img_path)
    # print(img.mode)
    w = img.width # 3600
    h = img.height # 3600
    if height is None:
        height = width
    if num_h is None:
        num_h = num_w
    y0 = 0 # 左上角坐标
    # 右下角坐标
    y1 = height
    for row in range(num_w):
        x0 = 0
        x1 = width
        for col in range(num_h):
            seg = img.crop((x0,y0,x1,y1))
            cnt = row * num_w + col + 1
            base_name = os.path.basename(img_path)
            seg_name =os.path.splitext(base_name)[0] + '_' + str(cnt) + os.path.splitext(base_name)[-1]
            seg.save(os.path.join(save_dir,seg_name))

            x0 = x1 - overlap_width
            if col == 0:
                x0 -= overlap_width
            x1 = x0 + width
        
        y0 = y1 - overlap_width
        if row == 0:
            y0 -= overlap_width
        y1 = y0 + height
    else:
        print('split_img over! ')



# def split_img(img_path:str, save_dir:str, num_w:int, num_h:int=None):
#     """
#     从左往右, 从上往下切图
#     """
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     img = Image.open(img_path)
#     # print(img.mode)
#     width= img.width # 3600
#     height = img.height # 3600
#     if num_h==None:
#         num_h = num_w
#     all_num = num_w * num_h
#     dx = dy = height // num_h
#     y0 = 0 # 左上角坐标
#     # 右下角坐标
#     y1 = dy

#     for row in range(num_w):
#         x0 = 0
#         x1 = dx
#         for col in range(num_h):
#             seg = img.crop((x0,y0,x1,y1))
#             cnt = row * num_w + col + 1
#             base_name = os.path.basename(img_path)
#             seg_name =os.path.splitext(base_name)[0] + '_' + str(cnt) + os.path.splitext(base_name)[-1]
#             seg.save(os.path.join(save_dir,seg_name),)

#             x0 = x1
#             x1 += dx
#         y0 = y1
#         y1 += dy
#     else:
#         print('split_img over! ',all_num,' images!')


if __name__ == '__main__':
    
        num_w = 15 # 3600*3600分成15 * 15个小图256*256有重叠
        sar_path = r'F:\dataset\SAR\sar.png'
        # path_gt = r'dataset\high_resolution_sar_image\sar_gt.png'
        path_gt_fined = r'F:\dataset\SAR\gt.png'
        # path_rgb = r'dataset\high_resolution_sar_image\sar_rgb.png'
        # width  = str(256) # 每个小图的宽
        sar_save = r"F:\dataset\SAR\256\sar_256"
        # save_gt = r'dataset\SAR_GT_' + width
        save_gt_fined = r"F:\dataset\SAR\256\gt_256"
        # split_img_with_overlap(path_gt_fined, save_gt_fined, 256, 15, 16)
        
        split_img_with_overlap(sar_path, sar_save, 256, 15, 16)
        # save_rgb = r"dataset\SAR_RGB_" + width
        # split_img(path,save,num_w)
        # split_img(path_gt, save_gt, num_w)
        # split_img(path_gt_fined,save_gt_fined,num_w)
        # split_img(path_rgb,save_rgb,num_w)


    # num_w = 2
    # path_mask = r'F:\dataset\Raw_AIR-PolarSAR-Seg\dataset\mask'
    # save_mask = r"F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\mask"
    # path_gt = r"F:\dataset\Raw_AIR-PolarSAR-Seg\dataset\gt"
    # save_gt = r"F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\gt"
    # path_images = r"F:\dataset\Raw_AIR-PolarSAR-Seg\dataset\images"
    # save_images = r"F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\images"

    # src = path_images
    # save = save_images
    # img_lst = os.listdir(src)
    # for img_name in img_lst:
    #     img_path = os.path.join(src, img_name)
    #     split_img(img_path, save, num_w)
    # pass
print("split over!")