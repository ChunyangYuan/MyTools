import xml.etree.cElementTree as ET
import os
import os.path as osp
import copy
import numpy as np
import cv2
import logging


# 创建名为 "local contrast" 的日志记录器
logger = logging.getLogger("local-contrast")

# 配置日志记录器的日志级别
logger.setLevel(logging.INFO)

# 创建一个处理程序，将日志消息发送到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个文件处理程序，将日志消息写入文件
file_handler = logging.FileHandler("local-contrast.log")
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

# 记录输出信息
# logger.info("This is an informational message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")


def local_contrast(img_folder: str, xml_folder: str) -> float:
    """
    local_contrast 计算一批图像数据的局部对比度

    Args:
        img_folder (str): 图片文件夹路径
        xml_folder (str): xml文件夹路径

    Returns:
        float: 一批图像数据的局部对比度
    """
    if not osp.exists(xml_folder):
        logger.info("xml_folder:{} do not exits!".format(xml_folder))
        sys.exit(1)
    if not osp.exists(xml_folder):
        logger.info("xml_folder:{} do not exits!".format(xml_folder))
        sys.exit(1)

    xml_list = sorted(os.listdir(xml_folder))
    img_list = sorted(os.listdir(img_folder))
    assert len(xml_list) == len(
        img_list), 'len(xml_list) and len(img_list) should be same!'

    local_contrast = 0
    # 没有目标的图像数量
    num_imgs_no_objects = 0
    for i in range(len(xml_list)):
        xml_path = osp.join(xml_folder, xml_list[i])
        img_path = osp.join(img_folder, img_list[i])
        # 计算当前图片中所有目标的局部对比度
        cur_local_contrast = calculate_local_contrast(img_path, xml_path)
        if cur_local_contrast == -1:
            num_imgs_no_objects += 1
            logger.info('There are no objects in the image {} '.format(
                img_list[i]))
        else:
            local_contrast += cur_local_contrast
            logger.info('local contrast value of the image {} is {}'.format(
                img_list[i], cur_local_contrast))
    local_contrast /= len(xml_list) - num_imgs_no_objects
    logger.info("local contrast: {:.4f}".format(local_contrast))
    return local_contrast


def calculate_local_contrast(image_path: str, xml_path: str) -> float:
    """
    calculate_local_contrast 计算当前图片中所有目标的局部对比度

    Args:
        image_path (str): 图像路径
        xml_path (str): xml文件路径

    Returns:
        float: 局部对比度
    """

    all_object_coordinates = get_all_object_coordinates(xml_path)
    # 判断目标坐标数组是否为空, 为空返回-1
    if not np.any(all_object_coordinates):
        return -1.0
    image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)  # height, width
    all_objects_local_contrast = []
    for bbox in all_object_coordinates:

        xmin, ymin, xmax, ymax = bbox
        # 计算目标最大灰度值
        max_gray_value = np.max(image[ymin:ymax, xmin:xmax])

        # 记录目标区域周围八个区域的均值
        region_means = []
        # 遍历目标周围的八个同样大小的区域
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                # 计算区域的坐标
                cur_region_xmin = xmin + i * (xmax - xmin)
                cur_region_ymin = ymin + j * (ymax - ymin)
                cur_region_xmax = xmax + i * (xmax - xmin)
                cur_region_ymax = ymax + j * (ymax - ymin)
                # 检查区域是否过界
                cur_region_xmin = max(0, cur_region_xmin)
                cur_region_ymin = max(0, cur_region_ymin)
                cur_region_xmax = min(image.shape[1]-1, cur_region_xmax)
                cur_region_ymax = min(image.shape[0]-1, cur_region_ymax)
                # 当前区域均值
                mean = np.mean(
                    image[cur_region_ymin:cur_region_ymax, cur_region_xmin:cur_region_xmax])
                region_means.append(mean)
        # 计算当前目标局部对比度
        cur_local_contrast = max_gray_value ** 2 / np.max(region_means)
        all_objects_local_contrast.append(cur_local_contrast)
    local_contrast = np.mean(all_objects_local_contrast)
    return local_contrast


def get_all_object_coordinates(xml_path: str) -> np.array:
    """
    get_all_object_coordinates 获取当前xml文件中所有目标的坐标信息并返回

    Args:
        xml_path (str): xml文件的路径

    Returns:
        np.array: 所有目标的坐标信息
    """
    # 解析 XML 文件
    tree = ET.parse(xml_path)

    # 获取根元素
    root = tree.getroot()
    coordinates = []
    # 遍历根元素的子元素
    for element in root:
        # 检查元素的标签名是否在目标元素列表中
        tag = element.tag
        if tag == 'object':
            for subelement in element:
                if subelement.tag == 'bndbox':
                    cur_coor = []  # [xmin,ymin,xmax,ymax]
                    for subsubelement in subelement:
                        cur_coor.append(int(subsubelement.text))
                    coordinates.append(cur_coor)
    return np.array(coordinates)


if __name__ == "__main__":
    image_folder = r'E:\dataset\SIRSTdevkit-master\local_contrast\PNGImages'
    xml_folder = r'E:\dataset\SIRSTdevkit-master\local_contrast\BBox'
    local_contrast(image_folder, xml_folder)
