import argparse
import os
import os.path as osp
import re
import sys
from PIL import Image
import cv2
import mmcv
import numpy as np


def crop_imgs_mmcv(single_img_path: str, output_folder: str, opt: argparse.ArgumentParser) -> None:
    """
    crop_imgs_mmcv crop_images using mmcv

    Args:
        single_img_path (str): single large image path.
        output_folder (str): output folder path to save sub-images.
        opt (argparse.ArgumentParser): options for crop_images.

    Raises:
        ValueError: only support 2 or 3 ndim images.

    Returns:
        _type_: None
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(single_img_path))

    img = mmcv.imread(single_img_path, flag='unchanged')

    if img.ndim == 2 or img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)

    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cv2.imwrite(
                osp.join(output_folder,
                         f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processed {img_name} successfully!!!'
    print(process_info)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare DIV2K dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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


def crop_munich_s1_img():
    args = parse_args()
    munich_s1 = r'F:\Dataset\multi_sensor_landcover_classification\images\Munich_s1.tif'
    output = r'F:\Dataset\multi_sensor_landcover_classification\output_folder\munich_s1'
    read_img(munich_s1)
    # extract subimages
    crop_imgs_mmcv(munich_s1, output, args)
    pass


def crop_munich_s1_anno():
    args = parse_args()
    munich_anno = r'F:\Dataset\multi_sensor_landcover_classification\annotations\munich_anno.tif'
    output = r'F:\Dataset\multi_sensor_landcover_classification\output_folder\munich_anno'
    read_img(munich_anno)
    # extract subimages
    crop_imgs_mmcv(munich_anno, output, args)
    pass


# a high-resolution SAR image that was acquired over Rosenehim by the TerraSAR-X satellite
def crop_Rosenehim_img():
    args = parse_args()
    sar_img = r'F:\Dataset\SAR\sar.png'
    output = r'F:\Dataset\SAR\output_folder\sar'
    # read_img(sar_img)
    # extract subimages
    crop_imgs_mmcv(sar_img, output, args)
    pass


def crop_Rosenehim_gt_img():
    args = parse_args()
    gt_img = r'F:\Dataset\SAR\gt.png'
    output = r'F:\Dataset\SAR\output_folder\gt'
    # read_img(gt_img)
    # extract subimages
    crop_imgs_mmcv(gt_img, output, args)
    pass


def crop_Rosenehim_color_gt_img():
    args = parse_args()
    color_gt_img = r'F:\Dataset\SAR\color_gt.png'
    output = r'F:\Dataset\SAR\output_folder\color_gt'
    # read_img(color_gt_img)
    # extract subimages
    crop_imgs_mmcv(color_gt_img, output, args)
    pass


if __name__ == '__main__':
    # crop_munich_s1_img()
    # crop_munich_s1_anno()
    # crop_Rosenehim_img()
    # crop_Rosenehim_gt_img()
    crop_Rosenehim_color_gt_img()
    pass
