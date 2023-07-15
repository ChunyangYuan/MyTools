import argparse
import os
import os.path as osp
import re
import sys
from PIL import Image
import cv2
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import copy


def test(img_path: str) -> None:
    img = mmcv.imread(img_path, flag='unchanged')  # bgr
    # img = mmcv.image.bgr2rgb(img)
    merged_img = np.zeros((500, 500, 3), dtype=np.uint8)
    merged_img[0:256, 0:256, ...] = img
    mmcv.imwrite(merged_img, 'test.png')
    plt.imshow(merged_img)  # rgb
    plt.axis('off')
    plt.show()


def merge_imgs_mmcv(imgs_path: str, output_folder: str, opt: argparse.ArgumentParser) -> None:
    """
    merge_imgs_mmcv merging all small images into a large image.

    Args:
        imgs_path (str): small images path
        output_folder (str): output dir
        opt (argparse.ArgumentParser): settings
    """
    if not osp.exists(output_folder):
        os.mkdir(output_folder)

    if not osp.exists(imgs_path):
        print("imgs_path do not exist! please check out!")
        sys.exit(1)

    origin_channels = opt['origin_channels']
    origin_width, origin_height = opt['origin_size']
    crop_size = opt['crop_size']
    thresh_size = opt['thresh_size']
    step = opt['step']

    # create canvas
    merged_img = np.zeros((origin_height, origin_width,
                          origin_channels), dtype=np.uint8)

    img_list = os.listdir(imgs_path)
    for i in range(len(img_list)):
        img_list[i] = osp.join(imgs_path, img_list[i])

    h_space = np.arange(0, origin_height - crop_size + 1, step)
    if origin_height - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, origin_height - crop_size)

    w_space = np.arange(0, origin_width - crop_size + 1, step)
    if origin_width - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, origin_width - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            img = mmcv.imread(img_list[index], flag='unchanged')
            index += 1
            merged_img[x:x + crop_size, y:y + crop_size, ...] = img

    merged_img_name = 'merged_' + osp.split(imgs_path)[-1] + '.png'
    cv2.imwrite(
        osp.join(output_folder, merged_img_name), merged_img,
        [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processed {merged_img_name} successfully!!!'
    print(process_info)


def parse_args(origin_size: tuple = (3600, 3600)):
    parser = argparse.ArgumentParser(
        description='Prepare DIV2K dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--origin_size',
        nargs='?',
        default=origin_size,
        type=tuple,
        help='merged large image size(w,h)')
    parser.add_argument(
        '--origin_channels',
        nargs='?',
        default=3,
        type=int,
        help='merged large image size')
    parser.add_argument(
        '--crop-size',
        nargs='?',
        default=256,
        type=int,
        help='cropped size for HR images')
    parser.add_argument(
        '--step',
        nargs='?',
        default=240,
        type=int,
        help='step size for HR images')
    parser.add_argument(
        '--thresh-size',
        nargs='?',
        default=0,
        type=int,
        help='threshold size for HR images')
    parser.add_argument(
        '--compression-level',
        nargs='?',
        default=3,
        type=int,
        help='compression level when save png images')

    args = parser.parse_args()
    for key, value in vars(args).items():
        print('%s: %s' % (key, value))
    return vars(args)


def read_img(file_path: str):
    img = Image.open(file_path)
    img = np.array(img)
    pass

# TODO ==============
# def merge_munich_s1_img():
#     size = (5596, 6031)
#     args = parse_args(size)
#     munich_s1 = r'F:\Dataset\multi_sensor_landcover_classification\images\Munich_s1.tif'
#     output = r'F:\Dataset\multi_sensor_landcover_classification\output_folder'
#     merge_imgs_mmcv(munich_s1, output, args)
#     pass

# TODO =====================
# def merge_munich_s1_anno():
#     size = (5596, 6031)
#     args = parse_args(size)
#     munich_anno = r'F:\Dataset\multi_sensor_landcover_classification\annotations\munich_anno.tif'
#     output = r'F:\Dataset\multi_sensor_landcover_classification\output_folder'
#     merge_imgs_mmcv(munich_anno, output, args)
#     pass

# TODO ===================================================
# # a high-resolution SAR image that was acquired over Rosenehim by the TerraSAR-X satellite
# def merge_Rosenehim_img():
#     size = (3600, 3600)
#     args = parse_args(size)
#     sar_img = r'F:\Dataset\SAR\output_folder\sar'
#     output = r'F:\Dataset\SAR\output_folder'
#     merge_imgs_mmcv(sar_img, output, args)
#     pass


def merge_Munich_classification_map():
    size = (5596, 6031)
    args = parse_args(size)
    gt_img = r'C:\Users\LwhYcy\Desktop\contrast_exp\classification_map\munich\munich_shgau'
    output = r'C:\Users\LwhYcy\Desktop\contrast_exp\classification_map\munich'
    merge_imgs_mmcv(gt_img, output, args)
    pass


def merge_Rosenehim_classification_map():
    size = (3600, 3600)
    args = parse_args(size)
    gt_img = r'C:\Users\LwhYcy\Desktop\contrast_exp\classification_map\sar\gcn_ss'
    output = r'C:\Users\LwhYcy\Desktop\contrast_exp\classification_map\sar'
    merge_imgs_mmcv(gt_img, output, args)
    pass


# def merge_Rosenehim_color_gt_img():
#     size = (3600, 3600)
#     args = parse_args(size)
#     color_gt_img = r'F:\Dataset\SAR\output_folder\color_gt'
#     output = r'F:\Dataset\SAR\output_folder'
#     merge_imgs_mmcv(color_gt_img, output, args)
#     test_equal()
#     pass


# def test_equal(color_gt: str = r'F:\Dataset\SAR\color_gt.png',
#                merged_color_gt: str = r'F:\Dataset\SAR\output_folder\merged_color_gt.png'):
#     gt = mmcv.imread(color_gt)
#     merged_gt = mmcv.imread(merged_color_gt)
#     print("gt.size={}".format(gt.size))
#     print(np.sum(gt == merged_gt))


# def find_img(input_folder: str = r'F:\Dataset\SAR\sar_output_folder\gt'):
#     img_list = os.listdir(input_folder)
#     for i in range(len(img_list)):
#         img_list[i] = osp.join(input_folder, img_list[i])
#     for img_path in img_list:
#         img = Image.open(img_path)
#         img = np.array(img)
#         if np.max(img) == 0:
#             print(img_path)


# def change_img_name(input_folder: str = r'C:\Users\LwhYcy\Desktop\contrast_exp\classification_map\MSSGU'):
#     img_name_list = os.listdir(input_folder)
#     new_name_list = copy.deepcopy(img_name_list)
#     for i in range(len(img_name_list)):
#         new_name = img_name_list[i][:8] + img_name_list[i][-4:]
#         img_name_list[i] = osp.join(input_folder, img_name_list[i])
#         new_name_list[i] = osp.join(input_folder, new_name)
#         os.rename(img_name_list[i], new_name_list[i])


# def mk_color_img():
#     # 创建一个大小为（256, 256）的RGB图像，初始颜色为黄色
#     width, height = 256, 256
#     color = (255, 255, 0)  # RGB颜色值为黄色
#     image_array = np.zeros((height, width, 3), dtype=np.uint8)
#     image_array[:, :] = color

#     # 从NumPy数组创建PIL图像对象
#     image = Image.fromarray(image_array)
#     image.save('yellow_image.png')

if __name__ == '__main__':

    # merge_Rosenehim_color_gt_img()

    merge_Rosenehim_classification_map()
    # merge_Munich_classification_map()

    pass
