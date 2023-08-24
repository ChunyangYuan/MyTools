import cv2
import numpy as np
import numpy
# import PIL.Image as Image
import os
import random
from typing import Tuple, Sequence, List, Union
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
import logging
from scipy.stats import multivariate_normal


# 创建名为 "ycy" 的日志记录器
logger = logging.getLogger("logger")

# 配置日志记录器的日志级别
logger.setLevel(logging.INFO)

# 创建一个处理程序,将日志消息发送到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个文件处理程序,将日志消息写入文件
file_handler = logging.FileHandler("dense_dataset.log")
file_handler.setLevel(logging.INFO)

# 创建一个格式化器,指定日志消息的格式
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# 为处理程序设置格式化器
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理程序添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.info("="*50)


def shrink_large_objects(image: np.ndarray,
                         mask: np.ndarray,
                         object_max_size: Union[int, Tuple[int, int]] = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    shrink_large_objects 缩小大目标

    Args:
        image (np.ndarray): 图像数据
        mask (np.ndarray): 掩码数据
        object_max_size (int, optional): 目标最大尺寸. Defaults to 7.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回处理后的图像和掩码
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if isinstance(object_max_size, int):
        max_height = max_width = object_max_size
    elif isinstance(object_max_size, Tuple) and len(object_max_size) == 2:
        max_width = object_max_size[0]
        max_height = object_max_size[1]
    for contour in contours:
        # Calculate the bounding rectangle for the contour
        original_object_x1, original_object_y1, original_object_w, original_object_h = cv2.boundingRect(
            contour)

        if original_object_w >= max_width or original_object_h >= max_height:

            # Calculate the scaling factor to shrink to target size
            scale_factor = min(
                max_width / original_object_w, max_height / original_object_h)

            # Calculate the new width and height
            new_w = int(original_object_w * scale_factor)
            new_h = int(original_object_h * scale_factor)

            # Calculate the new top-left corner
            new_x = int(original_object_x1 +
                        (original_object_w - new_w) / 2)
            new_y = int(original_object_y1 +
                        (original_object_h - new_h) / 2)

            # Crop the original region of interest and enlarge roi
            padding = 1
            img_height, img_width = image.shape[:2]

            roi_image = image[max(
                0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)]
            roi_mask = mask[max(
                0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)]

            # Resize the cropped region to target size
            resized_roi_image = cv2.resize(
                roi_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # roi_image2 = cv2.resize(
            #     resized_roi_image, (original_object_w+2*padding, original_object_h+2*padding), interpolation=cv2.INTER_LINEAR)
            # 定义腐蚀操作的结构元素（内核）
            # kernel_size = (7, 7)  # 调整内核的大小
            # # 调整内核的形状,也可以使用 cv2.MORPH_ELLIPSE 或 cv2.MORPH_CROSS cv2.MORPH_RECT
            # kernel_shape = cv2.MORPH_RECT

            # kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
            kernel = np.ones((5, 5), np.uint8)  # 5x5的正方形内核,可以根据需要调整大小

            # 进行腐蚀操作
            # eroded_image = cv2.erode(roi_image, kernel, iterations=1)
            kernel_size = (9, 9)
            # blurred = cv2.blur(roi_image, kernel_size)
            blurred = cv2.medianBlur(roi_image, ksize=11)
            # blurred = cv2.GaussianBlur(roi_image, kernel_size, 0)
            eroded_image = cv2.erode(blurred, kernel, iterations=1)
            # blurred_mask = cv2.GaussianBlur(roi_mask, kernel_size, 0)
            # blurred_mask[blurred_mask != 0] = 255

            # # blank_image = np.zeros_like(image)
            # result = cv2.bitwise_or(roi_image, roi_image2)
            resized_roi_mask = cv2.resize(
                roi_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_roi_mask[resized_roi_mask != 0] = 255

            # Fill the surrounding region in image with average pixel value of the eight surrounding boxes
            # avg_pixel_value = calculate_mean_around_box(
            #     image, original_object_x1, original_object_y1, original_object_w, original_object_h)
            # image[max(
            #     0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)] = avg_pixel_value
            # Update image with resized region
            image[max(
                0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)] = eroded_image
            # image[new_y:new_y + new_h, new_x:new_x + new_w] = blurred
            image[new_y:new_y + new_h, new_x:new_x +
                  new_w] = resized_roi_image
            # blurred = cv2.GaussianBlur(roi_image, kernel_size, 0)

            # Update mask with resized region and fill surrounding with 0
            mask[original_object_y1:original_object_y1+original_object_h,
                 original_object_x1:original_object_x1+original_object_w] = 0
            mask[new_y:new_y + new_h, new_x:new_x + new_w] = resized_roi_mask
            # mask[max(
            #     0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)] = blurred_mask
            # 增强图像目标
            # 定义亮度增加的值
            # brightness_increase = 50

            # # 使用布尔索引选择满足条件的像素位置
            # selected_pixels = mask == 255

            # # 在满足条件的位置上增加亮度,同时确保不超过255
            # # image[selected_pixels] = np.minimum(
            # #     image[selected_pixels] + brightness_increase, 230)
            # adjusted_pixels = image[selected_pixels] + brightness_increase
            # adjusted_pixels[adjusted_pixels > 255] = 255
            # image[selected_pixels] = adjusted_pixels

    return image, mask


def create_dense_dataset(image: np.ndarray,
                         mask: np.ndarray,
                         max_copies: int = 5,
                         max_offset: int = 15,
                         overlap: bool = False,
                         object_max_size: Union[int, Tuple[int, int]] = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    create_dense_dataset 根据给定的稀疏目标图像创建稠密目标图像

    Args:
        image (np.ndarray): 图像数据
        mask (np.ndarray): 掩码数据
        max_copies (int, optional): 最大的稠密目标数量. Defaults to 5.
        max_offset (int, optional): 最大偏差. Defaults to 15.
        overlap (bool) : 是否允许目标重叠. Defaults to False.
            注释：如果overlap是False，将采用高斯分布，进行无重叠密集目标的生成，且目标大小随机，保证不超过给定尺寸object_max_size。此外，如果想实现可能重叠且随机大小的密集目标的生成，可以将is_overlap置为False；

            如果是True，将使用类似于均匀分布的方式，进行可能重叠的密集目标的生成,此时只是进行相同目标多份复制，副本目标与原始目标尺寸大小一致。此时注意，原始目标如果大于给定尺寸object_max_size，那就无法保证副本目标的尺寸小于等于给定的object_max_size. 因此，可以先使用shrink_large_objects函数进行目标缩小，再使用该函数即可！
        object_max_size (Optional[int, Tuple[int, int]]) : int=7-> width = height = 7;
                                            Tuple[int,int]=(width,height). Defaults to 7.
    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回稠密目标的图像和掩码
    """
    dense_image = image.copy()
    dense_mask = mask.copy()

    # Find contours in the mask
    contours, _ = cv2.findContours(
        dense_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    group_gt_bboxes = []
    # cnt = 0
    gt_bboxes = []
    for contour in contours:
        # Calculate the bounding rectangle for the contour
        original_object_x1, original_object_y1, original_object_w, original_object_h = cv2.boundingRect(
            contour)
        # group bbox coordinate=(xmin,ymin,xmax,ymax)
        floating_window_x1 = max(0, original_object_x1 - max_offset)
        floating_window_y1 = max(0, original_object_y1 - max_offset)
        floating_window_x2 = min(
            dense_image.shape[1], original_object_x1 + original_object_w + max_offset)
        floating_window_y2 = min(
            dense_image.shape[0], original_object_y1 + original_object_h + max_offset)
        group_bbox = (floating_window_x1, floating_window_y1,
                      floating_window_x2, floating_window_y2)
        # 所有副本左上角和右下角坐标,初始化为当前目标
        all_objects_xmin = [original_object_x1]
        all_objects_ymin = [original_object_y1]
        all_objects_xmax = [original_object_x1+original_object_w]
        all_objects_ymax = [original_object_y1+original_object_h]
        # cnt += 1
        min_num_copies = 1
        min_offset = 7
        num_copies = random.randint(min_num_copies, max_copies)
        if overlap:
            for _ in range(num_copies):
                while True:
                    # Calculate random offsets for the copy
                    offset_x = random.randint(min_offset, max_offset)
                    if random.randint(-1, 0) == -1:
                        offset_x *= -1
                    offset_y = random.randint(min_offset, max_offset)
                    if random.randint(-1, 0) == -1:
                        offset_y *= -1

                    # Calculate the new top-left corner for the copy
                    new_x1 = max(0, original_object_x1 + offset_x)
                    new_y1 = max(0, original_object_y1 + offset_y)

                    # Calculate the new bottom-right corner for the copy
                    new_x2 = new_x1 + original_object_w
                    new_y2 = new_y1 + original_object_h

                    # Check if the new bottom-right corner is within image boundaries
                    if new_x2 <= dense_image.shape[1] and new_y2 <= dense_image.shape[0]:
                        # cnt += 1
                        all_objects_xmin.append(new_x1)
                        all_objects_ymin.append(new_y1)
                        all_objects_xmax.append(new_x2)
                        all_objects_ymax.append(new_y2)
                        # Calculate the dimensions of the overlapping region
                        overlap_w = new_x2 - new_x1
                        overlap_h = new_y2 - new_y1

                        # Crop and copy the overlapping region
                        roi_image = image[original_object_y1:original_object_y1 +
                                          overlap_h, original_object_x1:original_object_x1+overlap_w]
                        roi_mask = mask[original_object_y1:original_object_y1 +
                                        overlap_h, original_object_x1:original_object_x1+overlap_w]
                        dense_image[new_y1:new_y1 + overlap_h,
                                    new_x1:new_x1 + overlap_w] = roi_image
                        dense_mask[new_y1:new_y1 + overlap_h,
                                   new_x1:new_x1 + overlap_w] = roi_mask
                        break
        else:
            floating_window_width = floating_window_x2 - floating_window_x1
            floating_window_height = floating_window_y2 - floating_window_y1
            # generating normalized gauss matrix
            normalized_gauss_matrix = generate_gaussian_matrix(
                floating_window_height, floating_window_width)
            # # you can see gaussian matrix, if you remove the comments from the following code.
            # plt.imshow(normalized_gauss_matrix, cmap='viridis', origin='upper')
            # plt.colorbar()
            # plt.title('2D Gaussian Distribution Matrix')
            # plt.show()

            for i in range(num_copies):
                while True:
                    # 使用概率密度函数值进行不重复加权采样
                    idx = np.random.choice(floating_window_width * floating_window_height, replace=False,
                                           p=normalized_gauss_matrix.flatten())
                    # 将索引转换为左上角坐标，用于放置目标副本
                    new_y1 = idx // floating_window_height + floating_window_y1
                    new_x1 = idx % floating_window_width + floating_window_x1
                    # print('x={},y={}'.format(x, y))

                    # TODO resize原有目标来生成目标副本，保证不超过7像素，并计算目标的宽度和高度
                    # Calculate the scaling factor to shrink to target size
                    if isinstance(object_max_size, int):
                        max_height = max_width = object_max_size
                    elif isinstance(object_max_size, Tuple) and len(object_max_size) == 2:
                        max_width = object_max_size[0]
                        max_height = object_max_size[1]
                    target_width = random.randint(1, max_width)
                    target_height = random.randint(1, max_height)
                    scale_factor = min(
                        target_width / original_object_w, target_height / original_object_h)

                    # Calculate the new width and height
                    new_w = 1 if int(
                        original_object_w * scale_factor) == 0 else int(original_object_w * scale_factor)
                    new_h = 1 if int(
                        original_object_h * scale_factor) == 0 else int(original_object_h * scale_factor)

                    # Calculate the new bottom-right corner for the copy
                    new_x2 = new_x1 + new_w
                    new_y2 = new_y1 + new_h
                    # 确保物体不超出图像边界(new_x1,new_y已经保证了大于等于0)
                    if new_x2 > image.shape[1] or new_y2 > image.shape[0]:
                        continue
                    print('i:{}, new_h={}, new_w={}'.format(i+1, new_h, new_w))
                    # 检查物体是否与已有物体重叠，此外控制所有目标相邻距离>=margin
                    margin = 2
                    is_overlap = np.any(mask[max(new_y1-margin, 0):min(new_y2+margin, image.shape[0]),
                                             max(new_x1-margin, 0):min(new_x2+margin, image.shape[1])] == 255)
                    # 如果想实现可能重叠且随机大小的密集目标的生成，可以将is_overlap置为False,即取消下行代码注释使is_overlap = False, 直接通过if条件判断即可！
                    # is_overlap = False
                    if not is_overlap:
                        print('in non overlap..')
                        # Crop the original region of interest and shrink roi
                        padding = 0
                        img_height, img_width = image.shape[:2]

                        roi_image = image[max(
                            0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)]
                        # print(roi_image.shape)
                        roi_mask = mask[max(
                            0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)]
                        # resize image and mask
                        # cv2.INTER_LANCZOS4：Lanczos插值，使用像素网格周围的8x8像素来计算新像素的值，通常用于缩小图像。
                        # cv2.INTER_AREA：区域插值，根据图像区域的平均值来计算新像素的值，通常用于缩小图像。
                        resized_roi_image = cv2.resize(
                            roi_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        resized_roi_mask = cv2.resize(
                            roi_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        resized_roi_mask[resized_roi_mask != 0] = 255

                        # cv2.imshow('resized_roi_mask', resized_roi_mask)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        dense_image[new_y1:new_y1 + new_h, new_x1:new_x1 +
                                    new_w] = resized_roi_image
                        dense_mask[new_y1:new_y1 + new_h, new_x1:new_x1 +
                                   new_w] = resized_roi_mask

                        all_objects_xmin.append(new_x1)
                        all_objects_ymin.append(new_y1)
                        all_objects_xmax.append(new_x2)
                        all_objects_ymax.append(new_y2)
                        break
            pass
        # get final group bbox coordinate=(xmin,ymin,xmax,ymax)
        floating_window_x1 = max(floating_window_x1, min(all_objects_xmin))
        floating_window_y1 = max(floating_window_y1, min(all_objects_ymin))
        floating_window_x2 = min(floating_window_x2, max(all_objects_xmax))
        floating_window_y2 = min(floating_window_y2, max(all_objects_ymax))
        group_bbox = (floating_window_x1, floating_window_y1,
                      floating_window_x2, floating_window_y2)
        group_gt_bboxes.append(group_bbox)
        # gt_bboxes
        single_object_bbox = [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in zip(
            all_objects_xmin, all_objects_ymin, all_objects_xmax, all_objects_ymax)]
        gt_bboxes.extend(single_object_bbox)
    # logger.info('==================cnt"{}'.format(cnt))

    return dense_image, dense_mask, group_gt_bboxes, gt_bboxes


def generate_gaussian_matrix(rows, cols):
    center_row = rows // 2
    center_col = cols // 2

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    pos = np.dstack((x, y))
    center = np.array([center_col, center_row])

    covariance = np.eye(2) * (0.1 * max(rows, cols)) ** 2
    gauss_matrix = multivariate_normal.pdf(pos, mean=center, cov=covariance)
    # Normalize the Gaussian matrix
    normalized_gauss_matrix = gauss_matrix / np.sum(gauss_matrix)
    return normalized_gauss_matrix


def write_bbox_to_file(img_w: int,
                       img_h: int,
                       bboxes: List[Tuple[int, int, int, int]],
                       idx: str,
                       bbox_dir: str) -> None:
    """
    write_bbox_to_file create xml annotation file

    Args:
        img_w (int): image width
        img_h (int): image height
        bboxes (List[Tuple[int, int, int, int]]): bboxes coordinates
        idx (str): image name
        bbox_dir (str): the folder for saving xml file
    """

    annotation = ET.Element('annotation')
    file_name = ET.SubElement(annotation, 'file_name')
    file_name.text = idx

    _height, _width = img_h, img_w
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


# def save_plot_image(img: numpy.ndarray,
#                     bboxes: List[Tuple[int, int, int, int]],
#                     idx: str,
#                     output_dir: str,
#                     show: bool = False):
#     # 创建绘图对象
#     fig, ax = plt.subplots()
#     plt.axis('off')
#     plt.title(idx)
#     # 显示灰度图像
#     ax.imshow(img, cmap='gray', vmin=0, vmax=255)
#     for xmin, ymin, xmax, ymax in bboxes:
#         # 创建矩形框
#         rect = patches.Rectangle(
#             (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
#         # 将矩形框添加到图像上
#         ax.add_patch(rect)

#     save_fig_path = os.path.expanduser(os.path.join(output_dir, idx+'.png'))
#     plt.savefig(save_fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
#     # 显示图像, 手动关闭图片窗口, 程序才能继续执行
#     if show:
#         plt.show(block=False)  # 设置为非阻塞模式
#         # 设置定时器,延时1秒后关闭窗口
#         plt.pause(1)
#         plt.close()


def save_plot_image_cv2(img: np.ndarray,
                        bboxes: List[Tuple[int, int, int, int]],
                        idx: str,
                        output_dir: str,
                        show: bool = False):
    """
    save_plot_image_cv2 显示图像并自动关闭,再保存

    Args:
        img (np.ndarray): 图像数据
        bboxes (List[Tuple[int, int, int, int]]): 目标边界框的坐标列表,格式(xmin, ymin, xmax, ymax)左上右下角坐标
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

    # 显示图像1s,自动关闭
    if show:
        cv2.imshow('Image with Bounding Boxes', img_color)
        cv2.waitKey(0)  # show 1 s
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Folder paths
    # Provide the path to the image folder
    image_folder = r'dense\image'
    # Provide the path to the mask folder
    mask_folder = r'dense\mask'
    # Provide the path to the output folder
    output_folder = r'dense\Dense_IRSTD-1k\image'
    mask_output_folder = r'dense\Dense_IRSTD-1k\mask'
    group_anno_output = r'dense\Dense_IRSTD-1k\annotations_g'
    anno_output = r'dense\Dense_IRSTD-1k\annotations'
    visualize_group_gt = True
    visualize_gt = True
    show = True
    # Create the folders if they don't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)

    if not os.path.exists(group_anno_output):
        os.makedirs(group_anno_output)

    if not os.path.exists(anno_output):
        os.makedirs(anno_output)

    # Process images in the folder
    for file_name in os.listdir(image_folder):
        # Load image and mask
        image_path = os.path.join(image_folder, file_name)
        # Assuming mask filenames match image filenames
        mask_path = os.path.join(mask_folder, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Process large objects in the image and update the mask
        image, mask = shrink_large_objects(
            image.copy(), mask.copy())

        # 创建一个新窗口,并在窗口中显示两张图片
        # combined_image = cv2.hconcat([image, mask])
        # cv2.imshow('Combined Images', combined_image)
        # # 等待用户按下任意按键
        # cv2.waitKey(0)
        # # 关闭窗口
        # cv2.destroyAllWindows()

        # Generate a dense dataset using the new function
        dense_image, dense_mask, group_gt_bboxes, gt_bboxes = create_dense_dataset(
            image.copy(), mask.copy(), max_copies=5, max_offset=20)
        logger.info('filename:{}, cnt_group_gt_bboxes:{}, cnt_gt_bboxes:{}'.format(file_name,
                                                                                   len(group_gt_bboxes), len(gt_bboxes)))

        # Save the dense dataset(png format)
        file_name = os.path.splitext(file_name)[0]+'.png'
        output_image_path = os.path.join(output_folder, file_name)
        output_mask_path = os.path.join(
            mask_output_folder, file_name)
        cv2.imwrite(output_image_path, dense_image)
        cv2.imwrite(output_mask_path, dense_mask)

        # create single object xml annotation(with gt_bboxes)
        write_bbox_to_file(img_w=dense_image.shape[1], img_h=dense_image.shape[0],
                           bboxes=gt_bboxes, idx=os.path.splitext(file_name)[0], bbox_dir=anno_output)

        # create group xml(dense objects) annotation(with group_gt_bboxes)
        write_bbox_to_file(img_w=dense_image.shape[1], img_h=dense_image.shape[0],
                           bboxes=group_gt_bboxes, idx=os.path.splitext(file_name)[0], bbox_dir=group_anno_output)
        # # visualize and save for mask with gt_bboxes
        if visualize_gt:
            save_plot_image_cv2(
                img=dense_mask, bboxes=gt_bboxes, idx=os.path.splitext(file_name)[0]+'_mask', output_dir=anno_output, show=show)

        # visualize and save for mask with group_gt_bboxes
        if visualize_group_gt:
            save_plot_image_cv2(
                img=dense_mask, bboxes=group_gt_bboxes, idx=os.path.splitext(file_name)[0]+'_mask', output_dir=group_anno_output, show=show)

        # visualize and save for image with gt_bboxes
        if visualize_gt:
            save_plot_image_cv2(
                img=dense_image, bboxes=gt_bboxes, idx=os.path.splitext(file_name)[0], output_dir=anno_output, show=show)

        # visualize and save for image with group_gt_bboxes
        if visualize_group_gt:
            save_plot_image_cv2(
                img=dense_image, bboxes=group_gt_bboxes, idx=os.path.splitext(file_name)[0], output_dir=group_anno_output, show=show)

    # Display completion message
    logger.info('Processed completely.')
