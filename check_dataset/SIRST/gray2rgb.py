import cv2
import os


def convert_grayscale_to_rgb(input_folder, output_folder, image_list):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in image_list:
        # 构建输入和输出路径
        input_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)

        # 读取灰度图像
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 将灰度图像转换为RGB图像
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        # 保存RGB图像
        cv2.imwrite(output_path, rgb_image)
        print(f"保存文件 {image_name} 完成")


# 示例输入文件夹路径、输出文件夹路径和灰度图片列表
input_folder = 'E:\dataset\SIRSTdevkit-master\channel_1'
output_folder = 'E:\dataset\SIRSTdevkit-master\channel_3'

image_list = os.listdir(input_folder)

# 转换灰度图像为RGB图像并保存
convert_grayscale_to_rgb(input_folder, output_folder, image_list)
