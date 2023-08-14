import os
from PIL import Image


def resize_image(input_path, output_path, scale_factor=0.1):
    # 打开图片
    image = Image.open(input_path)

    # 获取原始图片的长宽
    original_width, original_height = image.size

    # 计算缩放后的新尺寸
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 缩放图片PIL.Image.Resampling.LANCZOS
    resized_image = image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS)

    # 保存缩放后的图片
    resized_image.save(output_path)


def process_images_in_folder(input_folder, output_folder, scale_factor=0.1):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, scale_factor)


# 调用示例
input_folder = r"C:\Users\LwhYcy\Desktop\classification_map\sar\map\segnet"
output_folder = r"C:\Users\LwhYcy\Desktop\classification_map\sar\reshape_map"
process_images_in_folder(input_folder, output_folder, scale_factor=0.1)
