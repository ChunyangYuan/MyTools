import cv2
import os
import numpy as np


def calculate_batch_stats(image_list):
    # 初始化 RGB 通道的均值和标准差
    mean_values = [0, 0, 0]
    std_values = [0, 0, 0]

    total_images = len(image_list)

    for image_path in image_list:
        # 读取图像
        image = cv2.imread(image_path)

        # 将图像转换为浮点型
        image_float = image.astype(np.float32)

        # 计算每个通道的均值和标准差
        mean_values += np.mean(image_float, axis=(0, 1))
        std_values += np.std(image_float, axis=(0, 1))

    # 计算均值和标准差
    mean_values /= total_images
    std_values /= total_images

    return mean_values, std_values


# 两个数据集图片路径列表
image_folder0 = r'E:\dataset\IRSTD-1k\IRSTD1k_Img'
image_folder1 = r'E:\dataset\SIRSTdevkit-master\PNGImages'
image_list0 = os.listdir(image_folder0)
image_list0 = [os.path.join(image_folder0, img_name)
               for img_name in image_list0]
image_list1 = os.listdir(image_folder1)
image_list1 = [os.path.join(image_folder1, img_name)
               for img_name in image_list1]
image_list0.extend(image_list1)
image_list = image_list0
# 计算整个一批图像的 RGB 通道的均值和标准差
mean_values, std_values = calculate_batch_stats(image_list)

# 格式化输出
output_mean = f"均值 = [{mean_values[2]:.4f}, {mean_values[1]:.4f}, {mean_values[0]:.4f}]"
output_std = f"标准差 = [{std_values[2]:.4f}, {std_values[1]:.4f}, {std_values[0]:.4f}]"

# 打印结果
print(output_mean)
print(output_std)
