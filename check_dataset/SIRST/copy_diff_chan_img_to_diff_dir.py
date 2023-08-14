import os
import shutil
import cv2


def count_channels(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(image.shape) < 3:
        return 1  # 单通道图像
    else:
        return image.shape[-1]  # 多通道图像的通道数


def copy_images_by_channels(input_folder, output_folder_single, output_folder_three, output_folder_four):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder_single, exist_ok=True)
    os.makedirs(output_folder_three, exist_ok=True)
    os.makedirs(output_folder_four, exist_ok=True)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)

        channels = count_channels(image_path)

        # 根据通道数将图像复制到相应的输出文件夹
        if channels == 1:
            output_path = os.path.join(output_folder_single, filename)
            shutil.copyfile(image_path, output_path)
        elif channels == 3:
            output_path = os.path.join(output_folder_three, filename)
            shutil.copyfile(image_path, output_path)
        elif channels == 4:
            output_path = os.path.join(output_folder_four, filename)
            shutil.copyfile(image_path, output_path)


input_folder = r'E:\dataset\SIRSTdevkit-master\PNGImages'
output_folder_single = r'E:\dataset\SIRSTdevkit-master\channel_1'
output_folder_three = r'E:\dataset\SIRSTdevkit-master\channel_3'
output_folder_four = r'E:\dataset\SIRSTdevkit-master\channel_4'

copy_images_by_channels(input_folder, output_folder_single,
                        output_folder_three, output_folder_four)
