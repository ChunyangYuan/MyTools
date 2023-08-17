"""Convert gt_seg_maps to gt_bboxes"""
import os
import mmcv
import skimage.measure as skm
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from lxml import etree
import logging
import matplotlib.patches as patches
import time
from typing import Tuple, List
import cv2


enlarge_flag = True
padding = 0
visualize = True
data_root = 'E:\dataset\IRSTD-1k'
idx_file = os.path.join(data_root, 'splits', 'test_.txt')
img_dir = os.path.join(data_root, 'IRSTD1k_Img')
mask_dir = os.path.join(data_root, 'IRSTD1k_Label')
bbox_dir = os.path.join(data_root, 'BBox_test')
if not os.path.exists(bbox_dir):
    os.makedirs(bbox_dir)


def save_plot_image_cv2(img: np.ndarray,
                        bboxes: List[Tuple[int]],
                        idx: str,
                        output_dir: str,
                        show: bool = False):
    """
    save_plot_image_cv2 显示图像并自动关闭，再保存(使用可视化会慢一些，
                        如果只想保存可视化结果，不显示的话，可以手动修改show=False)

    Args:
        img (np.ndarray): 图像数据
        bboxes (List[Tuple[int]]): 目标边界框的坐标列表，格式(xmin, ymin, xmax, ymax)左上右下角坐标
        idx (str): 图片名称(无扩展名)
        output_dir (str): 图片保存路径
        show (bool, optional): 是否可视化. Defaults to False.
    """
    # 绘制矩形框的颜色 (红色)
    rect_color = (0, 0, 255)
    # 转换灰度图像为彩色图像（3通道）
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for xmin, ymin, xmax, ymax in bboxes:
        # 在图像上绘制矩形框
        cv2.rectangle(img_color, (xmin, ymin), (xmax, ymax), rect_color, 1)

    save_fig_path = os.path.expanduser(os.path.join(output_dir, idx+'.png'))
    cv2.imwrite(save_fig_path, img_color)

    # 显示图像, 手动关闭图片窗口, 程序才能继续执行
    if show:
        cv2.imshow('Image with Bounding Boxes', img_color)
        cv2.waitKey(1000)  # show 1 s
        cv2.destroyAllWindows()


def write_bbox_to_file(img: np.ndarray, bboxes: List[Tuple[int]], idx: str) -> None:
    """
    write_bbox_to_file 将目标框及图像其他信息写入xml文件

    Args:
        img (np.ndarray): 图像数据
        bboxes (List[Tuple[int]]): 目标框列表，格式(xmin, ymin, xmax, ymax)左上右下角坐标
        idx (str): 图像名称(无扩展名)
    """

    annotation = ET.Element('annotation')
    filename = ET.SubElement(annotation, 'filename')
    filename.text = idx

    _height, _width = img.shape
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(_width)
    height = ET.SubElement(size, 'height')
    height.text = str(_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(1)

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        label_object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(label_object, 'name')
        name.text = 'Target'

        _bbox = ET.SubElement(label_object, 'bndbox')
        xmin_elem = ET.SubElement(_bbox, 'xmin')
        xmin_elem.text = str(xmin)

        ymin_elem = ET.SubElement(_bbox, 'ymin')
        ymin_elem.text = str(ymin)

        xmax_elem = ET.SubElement(_bbox, 'xmax')
        xmax_elem.text = str(xmax)

        ymax_elem = ET.SubElement(_bbox, 'ymax')
        ymax_elem.text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree_str = ET.tostring(tree.getroot(), encoding='unicode')
    save_xml_path = os.path.join(bbox_dir, idx + '.xml')
    root = etree.fromstring(tree_str).getroottree()
    root.write(save_xml_path, pretty_print=True)


def main():
    # load image and mask paths
    with open(idx_file, "r") as lines:
        # logging.info("lines:", lines)
        for line in lines:
            idx = line.rstrip('\n')
            logging.info(idx)
            _image = os.path.join(img_dir, idx + ".png")
            _mask = os.path.join(mask_dir, idx + ".png")

            img = mmcv.imread(_image, flag='grayscale')
            hei, wid = img.shape[:2]
            mask = mmcv.imread(_mask, flag='grayscale')
            label_img = skm.label(mask, background=0)
            regions = skm.regionprops(label_img)
            bboxes = []
            for region in regions:
                # convert starting from 0 to starting from 1
                ymin, xmin, ymax, xmax = np.array(region.bbox)

                if enlarge_flag:
                    ymin -= padding
                    xmin -= padding
                    xmax += padding
                    ymax += padding

                # ymin, xmin are inside the region, but xmax and ymax not
                # e.g., bbox[ymin,xmin, ymax,xmax] = [0,0,20,20] -> [0,0,19,19] included
                # ymin -= 1
                # xmin -= 1
                # boundary check
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(wid, xmax)
                ymax = min(hei, ymax)
                bboxes.append([xmin, ymin, xmax, ymax])
            logging.info(bboxes)
            # visualize
            if visualize:
                save_plot_image_cv2(
                    img, bboxes, idx, output_dir=bbox_dir, show=True)

            # create xml for bboxes
            write_bbox_to_file(img, bboxes, idx)
            # break


if __name__ == '__main__':
    main()
