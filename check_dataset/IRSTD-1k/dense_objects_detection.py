import xml.etree.cElementTree as ET
import os
import os.path as osp
import copy
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure
from scipy.ndimage import gaussian_laplace


def extract_target_bbox(xml_path: str) -> List[Tuple[int, int, int, int]]:
    """
    extract_target_bbox 从xml注释文件中提取组框坐标(一群密集小组目标成一个组，
                        一个组一个坐标框，由左上角坐标和右下角坐标组成)

    Args:
        xml_path (str): xml文件路径

    Returns:
        List[Tuple[int, int, int, int]]: 返回所有组框坐标
            bbox format:[(xmin1, ymin1, xmax1, ymax1),(xmin2, ymin2, xmax2, ymax2),...]
    """

    # 解析 XML 文件
    tree = ET.parse(xml_path)

    # 获取根元素
    root = tree.getroot()
    bboxes = []
    # 遍历根元素的子元素
    for element in root:
        # 检查元素的标签名是否在目标元素列表中
        tag = element.tag
        if tag == 'object':
            for subelement in element:
                if subelement.tag == 'bndbox':
                    bbox = []
                    for subsubelement in subelement:
                        bbox.append(int(subsubelement.text))
                    bboxes.append(bbox)
    # print(bboxes)
    return bboxes


def dense_objects_detection(img: str,
                            bboxes: List[List[int]]
                            ) -> List[Tuple[int, int, int, int]]:
    """
    dense_objects_detection 从带有组目标坐标框(一群密集小目标组成一个组，
        一个组一个坐标框，由左上角坐标和右下角坐标组成)的预测结果中检测所有单个小目标并返回其坐标

    Args:
        img (str): 带有组目标坐标框的预测结果
        bboxes (List[List[int]]): 组目标坐标框
            bbox format:[[xmin1, ymin1, xmax1, ymax1],[xmin2, ymin2, xmax2, ymax2],...]

    Returns:
        List[Tuple[int, int, int, int]]: 所有单个小目标在原始图像上的绝对坐标
            bbox format:[(xmin1, ymin1, xmax1, ymax1),(xmin2, ymin2, xmax2, ymax2),...]
    """
    all_objects_coordinates = []
    padding = 3
    # 读取灰度图像
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    for xmin, ymin, xmax, ymax in bboxes:
        # 添加padding，并确保坐标在图像范围内
        xmin = max(0, xmin-padding)
        ymin = max(0, ymin-padding)
        xmax = min(image.shape[1], xmax+padding)
        ymax = min(image.shape[0], ymax+padding)

        # 提取目标区域并添加到列表
        target_region = image[ymin:ymax, xmin:xmax]
        # 提取目标区域中的所有目标坐标并添加到列表objects_coordinates
        object_relative_coordinates = blob_detection(target_region)
        # 映射小目标的相对坐标到原始图像的绝对坐标
        object_absolute_coordinates = []
        for target_xmin, target_ymin, target_xmax, target_ymax in object_relative_coordinates:
            absolute_coordinate = (
                target_xmin + xmin,
                target_ymin + ymin,
                target_xmax + xmin,
                target_ymax + ymin
            )
            object_absolute_coordinates.append(absolute_coordinate)

        all_objects_coordinates.extend(object_absolute_coordinates)

    return all_objects_coordinates


def blob_detection(roi_image: np.ndarray,
                   method: str = 'DOG',
                   show: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    blob_detection 斑点检测算法，即从给定的目标区域检测出所有物体（称为斑点）的相对坐标（相对于原始图像来讲）,显示图像

    Args:
        roi_image (np.ndarray): 目标区域
        method (str, optional): 斑点检测算法,只支持两种算法，即'DOG'和'LOG'. Defaults to 'DOG'.
        show (str) : 是否可视化提取目标的图像. Defaults to False.

    Raises:
        ValueError: 只支持两种算法，即'DOG'和'LOG', 输入其他方法字符串，则抛出异常

    Returns:
        List[Tuple[int, int, int, int]]: 目标区域中所有单个小目标的相对坐标（相对于原始图像来讲）
            bbox format:[(xmin1, ymin1, xmax1, ymax1),(xmin2, ymin2, xmax2, ymax2),...]
    """

    # 图像如果不是灰度图像，将其转换为灰度图像
    if len(roi_image.shape) > 2:
        gray_image = color.rgb2gray(roi_image)
    else:
        gray_image = roi_image
    if method == 'LOG':
        # 使用Laplacian of Gaussian (LoG) 方法
        edges = gaussian_laplace(gray_image, sigma=1)
        # 标记连通组件并获取边界框
        label_image, num_labels = measure.label(
            edges > 0, connectivity=2, return_num=True)
        regions = measure.regionprops(label_image)
        # 保存目标坐标到列表中
        object_relative_coordinates = [(region.bbox[1], region.bbox[0], region.bbox[3],
                                        region.bbox[2]) for region in regions]
        pass
    elif method == 'DOG':
        # 使用Difference of Gaussian (DoG) 方法
        edges = filters.difference_of_gaussians(
            gray_image, low_sigma=1, high_sigma=3)
        # 标记连通组件并获取边界框
        label_image, num_labels = measure.label(
            edges > 0, connectivity=2, return_num=True)
        regions = measure.regionprops(label_image)
        # 保存目标坐标到列表中
        object_relative_coordinates = [(region.bbox[1], region.bbox[0], region.bbox[3],
                                        region.bbox[2]) for region in regions]
    else:
        raise ValueError("method only supports 'DOG' and 'LOG'!")
    # 可视化
    if show:
        # 显示原始图像、灰度图像以及处理后的图像和边界框
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(roi_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(gray_image, cmap='gray')
        axes[1].set_title('Grayscale Image')
        axes[1].axis('off')

        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title('Edges using '+method)
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor='red', linewidth=1)
            axes[2].add_patch(rect)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show(block=False)  # 设置为非阻塞模式

        # 设置定时器，延时2秒后关闭窗口
        plt.pause(5)
        plt.close()

    # 返回提取的目标坐标列表
    return object_relative_coordinates


if __name__ == "__main__":
    xml = r'data\test\group\XDU1.xml'
    image = r'XDU1_mask.png'
    bboxes = extract_target_bbox(xml)
    all_objects_coordinates = dense_objects_detection(image, bboxes)
    print(all_objects_coordinates)
