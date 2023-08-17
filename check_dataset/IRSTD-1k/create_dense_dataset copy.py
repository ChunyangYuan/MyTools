import cv2
import numpy as np
import numpy
# import PIL.Image as Image
import os
import random
from typing import Tuple, Sequence, List
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

# 创建名为 "ycy" 的日志记录器
logger = logging.getLogger("logger")

# 配置日志记录器的日志级别
logger.setLevel(logging.INFO)

# 创建一个处理程序，将日志消息发送到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个文件处理程序，将日志消息写入文件
file_handler = logging.FileHandler("dense_dataset.log")
file_handler.setLevel(logging.INFO)

# 创建一个格式化器，指定日志消息的格式
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
                         target_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    shrink_large_objects 缩小大目标

    Args:
        image (np.ndarray): 图像数据
        mask (np.ndarray): 掩码数据
        target_size (int, optional): 目标最大尺寸. Defaults to 7.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回处理后的图像和掩码
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        if w > target_size or h > target_size:
            # Calculate the scaling factor to shrink to target size
            scale_factor = min(target_size / w, target_size / h)

            # Calculate the new width and height
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)

            # Calculate the new top-left corner
            new_x = int(x + (w - new_w) / 2)
            new_y = int(y + (h - new_h) / 2)

            # Crop the original region of interest and enlarge roi
            padding = 5
            img_height, img_width = image.shape[:2]

            roi_image = image[max(
                0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)]
            roi_mask = mask[max(
                0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)]

            # Resize the cropped region to target size
            resized_roi_image = cv2.resize(
                roi_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # roi_image2 = cv2.resize(
            #     resized_roi_image, (w+2*padding, h+2*padding), interpolation=cv2.INTER_LINEAR)
            # 定义腐蚀操作的结构元素（内核）
            # kernel_size = (7, 7)  # 调整内核的大小
            # # 调整内核的形状，也可以使用 cv2.MORPH_ELLIPSE 或 cv2.MORPH_CROSS cv2.MORPH_RECT
            # kernel_shape = cv2.MORPH_RECT

            # kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
            kernel = np.ones((5, 5), np.uint8)  # 5x5的正方形内核，可以根据需要调整大小

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
            #     image, x, y, w, h)
            # image[max(
            #     0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)] = avg_pixel_value
            # Update image with resized region
            image[max(
                0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)] = eroded_image
            # image[new_y:new_y + new_h, new_x:new_x + new_w] = blurred
            image[new_y:new_y + new_h, new_x:new_x + new_w] = resized_roi_image
            # blurred = cv2.GaussianBlur(roi_image, kernel_size, 0)

            # Update mask with resized region and fill surrounding with 0
            mask[y:y+h, x:x+w] = 0
            mask[new_y:new_y + new_h, new_x:new_x + new_w] = resized_roi_mask
            # mask[max(
            #     0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)] = blurred_mask
            # 增强图像目标
            # 定义亮度增加的值
            # brightness_increase = 50

            # # 使用布尔索引选择满足条件的像素位置
            # selected_pixels = mask == 255

            # # 在满足条件的位置上增加亮度，同时确保不超过255
            # # image[selected_pixels] = np.minimum(
            # #     image[selected_pixels] + brightness_increase, 230)
            # adjusted_pixels = image[selected_pixels] + brightness_increase
            # adjusted_pixels[adjusted_pixels > 255] = 255
            # image[selected_pixels] = adjusted_pixels

    return image, mask


def calculate_mean_around_box(image: np.ndarray,
                              x: int,
                              y: int,
                              width: int,
                              height: int) -> numpy.uint8:
    """
    calculate_mean_around_box 以给定坐标框为中心，计算其八邻域像素均值

    Args:
        image (np.ndarray): 图像数据
        x (int): 左上角x坐标
        y (int): 左上角y坐标
        width (int): 目标框宽度
        height (int): 目标框高度

    Returns:
        numpy.uint8: 返回八邻域像素均值
    """
    # Calculate the coordinates of the eight surrounding boxes
    boxes = [
        (x - width, y - height, x, y),             # 左上角
        (x, y - height, x + width, y),             # 上
        (x + width, y - height, x + width * 2, y),  # 右上角
        (x - width, y, x, y + height),             # 左
        (x + width, y, x + width * 2, y + height),  # 右
        (x - width, y + height, x, y + height * 2),  # 左下角
        (x, y + height, x + width, y + height * 2),  # 下
        (x + width, y + height, x + width * 2, y + height * 2)  # 右下角
    ]

    # Ensure box boundaries are within the image dimensions
    img_height, img_width = image.shape[:2]
    boxes = [(max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2))
             for x1, y1, x2, y2 in boxes]

    means = []
    for box in boxes:
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        mean = np.mean(roi)
        means.append(mean)

    overall_mean = np.mean(means).astype(np.uint8)
    return overall_mean


def create_dense_dataset(image: np.ndarray,
                         mask: np.ndarray,
                         max_copies: int = 7,
                         max_offset: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    create_dense_dataset 根据给定的稀疏目标图像创建稠密目标图像

    Args:
        image (np.ndarray): 图像数据
        mask (np.ndarray): 掩码数据
        max_copies (int, optional): 最大的稠密目标数量. Defaults to 7.
        max_offset (int, optional): 最大偏差. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回稠密目标的图像和掩码
    """
    dense_image = image.copy()
    dense_mask = mask.copy()
    min_num_copies = 3

    min_offset = 7
    # Find contours in the mask
    contours, _ = cv2.findContours(
        dense_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    group_bboxs = []
    # cnt = 0
    single_object_bboxs = []
    for contour in contours:
        # Calculate the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)
        # group bbox coordinate=(xmin,ymin,xmax,ymax)
        xmin_g = max(0, x - max_offset)
        ymin_g = max(0, y - max_offset)
        xmax_g = min(dense_image.shape[1], x + w + max_offset)
        ymax_g = min(dense_image.shape[0], y + h + max_offset)
        group_bbox = (xmin_g, ymin_g, xmax_g, ymax_g)
        # 所有副本左上角和右下角坐标,初始化为当前目标
        all_xmin = [x]
        all_ymin = [y]
        all_xmax = [x+w]
        all_ymax = [y+h]
        # cnt += 1
        num_copies = random.randint(min_num_copies, max_copies)
        for _ in range(num_copies):

            # Calculate random offsets for the copy
            offset_x = random.randint(min_offset, max_offset)
            if random.randint(-1, 0) == -1:
                offset_x *= -1
            offset_y = random.randint(min_offset, max_offset)
            if random.randint(-1, 0) == -1:
                offset_y *= -1

            # Calculate the new top-left corner for the copy
            new_x = max(0, x + offset_x)
            new_y = max(0, y + offset_y)

            # Calculate the new bottom-right corner for the copy
            new_x2 = new_x + w
            new_y2 = new_y + h

            # Check if the new bottom-right corner is within image boundaries
            if new_x2 <= dense_image.shape[1] and new_y2 <= dense_image.shape[0]:
                # cnt += 1
                all_xmin.append(new_x)
                all_ymin.append(new_y)
                all_xmax.append(new_x2)
                all_ymax.append(new_y2)
                # Calculate the dimensions of the overlapping region
                overlap_w = new_x2 - new_x
                overlap_h = new_y2 - new_y

                # Crop and copy the overlapping region
                roi_image = image[y:y+overlap_h, x:x+overlap_w]
                roi_mask = mask[y:y+overlap_h, x:x+overlap_w]
                dense_image[new_y:new_y + overlap_h,
                            new_x:new_x + overlap_w] = roi_image
                dense_mask[new_y:new_y + overlap_h,
                           new_x:new_x + overlap_w] = roi_mask
        # get final group bbox coordinate=(xmin,ymin,xmax,ymax)
        xmin_g = max(xmin_g, min(all_xmin))
        ymin_g = max(ymin_g, min(all_ymin))
        xmax_g = min(xmax_g, max(all_xmax))
        ymax_g = min(ymax_g, max(all_ymax))
        group_bbox = (xmin_g, ymin_g, xmax_g, ymax_g)
        group_bboxs.append(group_bbox)
        # single_object_bboxs
        single_object_bbox = [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in zip(
            all_xmin, all_ymin, all_xmax, all_ymax)]
        single_object_bboxs.extend(single_object_bbox)
    # logger.info('==================cnt"{}'.format(cnt))

    return dense_image, dense_mask, group_bboxs, single_object_bboxs


def write_bbox_to_file(img_w: int,
                       img_h: int,
                       bboxes: List[Tuple[int]],
                       idx: str,
                       bbox_dir: str) -> None:
    """
    write_bbox_to_file create xml annotation file

    Args:
        img_w (int): image width
        img_h (int): image height
        bboxes (List[Tuple[int]]): bboxs coordinates
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


def save_plot_image(img: numpy.ndarray,
                    bboxes: List[Tuple[int]],
                    idx: str,
                    output_dir: str,
                    show: bool = False):
    # 创建绘图对象
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.title(idx)
    # 显示灰度图像
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    for xmin, ymin, xmax, ymax in bboxes:
        # 创建矩形框
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        # 将矩形框添加到图像上
        ax.add_patch(rect)

    save_fig_path = os.path.expanduser(os.path.join(output_dir, idx+'.png'))
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # 显示图像, 手动关闭图片窗口, 程序才能继续执行
    if show:
        plt.show()
        plt.close()


def save_plot_image_cv2(img: np.ndarray,
                        bboxes: List[Tuple[int]],
                        idx: str,
                        output_dir: str,
                        show: bool = False):
    """
    save_plot_image_cv2 显示图像并自动关闭，再保存

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


if __name__ == '__main__':
    # Folder paths
    image_folder = r'data/image'  # Provide the path to the image folder
    mask_folder = r'data/mask'    # Provide the path to the mask folder
    output_folder = r'data/results/images'  # Provide the path to the output folder
    mask_output_folder = r'data/results/masks'
    group_xml_output = r'data/results/annotations_g'
    xml_output = r'data/results/annotations'
    visualize_group_gt = True
    visualize_gt = True
    # Create the folders if they don't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)

    if not os.path.exists(group_xml_output):
        os.makedirs(group_xml_output)

    if not os.path.exists(xml_output):
        os.makedirs(xml_output)

    # Process images in the folder
    for file_name in os.listdir(image_folder):
        # Load image and mask
        image_path = os.path.join(image_folder, file_name)
        # Assuming mask filenames match image filenames
        mask_path = os.path.join(mask_folder, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Process large objects in the image and update the mask
        processed_image, processed_mask = shrink_large_objects(
            image.copy(), mask.copy())

        # Generate a dense dataset using the new function
        dense_image, dense_mask, group_bboxs, single_object_bboxs = create_dense_dataset(
            processed_image.copy(), processed_mask.copy(), max_copies=7, max_offset=10)
        logger.info('filename:{}, cnt_group_bboxs:{}, cnt_single_object_bboxs:{}'.format(file_name,
                                                                                         len(group_bboxs), len(single_object_bboxs)))

        # Save the dense dataset(png format)
        file_name = os.path.splitext(file_name)[0]+'.png'
        output_image_path = os.path.join(output_folder, file_name)
        output_mask_path = os.path.join(
            mask_output_folder, file_name)
        cv2.imwrite(output_image_path, dense_image)
        cv2.imwrite(output_mask_path, dense_mask)

        # create single object xml annotation
        write_bbox_to_file(img_w=dense_image.shape[1], img_h=dense_image.shape[0],
                           bboxes=single_object_bboxs, idx=os.path.splitext(file_name)[0], bbox_dir=xml_output)
        # visualize and save
        if visualize_gt:
            save_plot_image_cv2(
                img=dense_image, bboxes=single_object_bboxs, idx=os.path.splitext(file_name)[0], output_dir=xml_output, show=True)

        # create group xml(dense objects) annotation
        write_bbox_to_file(img_w=dense_image.shape[1], img_h=dense_image.shape[0],
                           bboxes=group_bboxs, idx=os.path.splitext(file_name)[0], bbox_dir=group_xml_output)
        # visualize and save
        if visualize_group_gt:
            save_plot_image(
                img=dense_mask, bboxes=group_bboxs, idx=os.path.splitext(file_name)[0], output_dir=group_xml_output, show=True)

    # Display completion message
    logger.info('Processing complete.')
