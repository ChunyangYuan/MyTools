import PIL.Image as Image
import numpy as np
import logging


# 创建名为 "local contrast" 的日志记录器
logger = logging.getLogger("classes-count")

# 配置日志记录器的日志级别
logger.setLevel(logging.INFO)

# 创建一个处理程序，将日志消息发送到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个文件处理程序，将日志消息写入文件
file_handler = logging.FileHandler("classes-count.log")
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

train_ratio = 0.05
val_ratio = 0.01


def calc_classes_count_sar(gt_path: str = r'F:\Dataset\SAR\gt.png'):
    img = Image.open(gt_path)
    img = np.array(img)
    # class_count = [0 for _ in range(np.unique(img))]
    classes_label = ['unknown', 'water', 'forest', 'building', 'farmland']
    logger.info('sar')
    total = 0
    for i in range(len(np.unique(img))):
        count = np.sum(img == i)
        total += count
        train_num = np.floor(count*train_ratio).astype('int32')
        val_num = np.floor(count*val_ratio).astype('int32')
        test_num = count - train_num - val_num
        logger.info('label:{} -> ( train:{}, val:{}, test:{}, total:{} )'.format(
            classes_label[i], train_num, val_num, test_num, count))
    logger.info('Total:{}'.format(total))


def calc_classes_count_munich(gt_path: str = r'F:\Dataset\multi_sensor_landcover_classification\annotations\munich_anno.tif'):
    img = Image.open(gt_path)
    img = np.array(img)
    # class_count = [0 for _ in range(np.unique(img))]
    classes_label = ['unknown', 'agriculture field',
                     'forest', 'built-up', 'water body']
    logger.info('munich dataset')
    total = 0
    for i in range(len(np.unique(img))):
        count = np.sum(img == i)
        total += count
        train_num = np.floor(count*train_ratio).astype('int32')
        val_num = np.floor(count*val_ratio).astype('int32')
        test_num = count - train_num - val_num
        logger.info('label:{} -> ( train:{}, val:{}, test:{}, total:{} )'.format(
            classes_label[i], train_num, val_num, test_num, count))
    logger.info('Total:{}'.format(total))


calc_classes_count_sar()
calc_classes_count_munich()
