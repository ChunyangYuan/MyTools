import os
import shutil


def copy_images(source_folder, destination_folder, image_names):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for image_name in image_names:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copyfile(source_path, destination_path)
        print(f"复制文件 {image_name} 完成")


# 示例图片文件夹路径和输出文件夹路径
source_folder = 'E:\dataset\SIRSTdevkit-master\PNGImages_2X'
destination_folder = 'E:\dataset\SIRSTdevkit-master\Misc_ch3_2x'

# 示例图片名称列表
input_dir = r'E:\dataset\SIRSTdevkit-master\Misc_ch3'
image_names = os.listdir(input_dir)
# image_names = [os.path.splitext(img)[0]+'.xml' for img in image_names]

# 复制图片到输出文件夹
copy_images(source_folder, destination_folder, image_names)
