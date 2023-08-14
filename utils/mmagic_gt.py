from PIL import Image
import os


def get_image_shape(image_path):
    img = Image.open(image_path)
    return img.size


def write_to_txt(image_folder, output_txt):
    with open(output_txt, 'w') as file:
        for filename in sorted(os.listdir(image_folder)):
            if filename.endswith(".png"):
                image_path = os.path.join(image_folder, filename)
                shape = get_image_shape(image_path)
                print(f"{filename} {shape}\n")
                file.write(f"{filename} {shape}\n")


if __name__ == "__main__":
    image_folder_path = "F:\PythonProjects\mmagic-main\data\Misc\misc"  # 替换成你的图像文件夹路径
    output_txt_file = "F:\PythonProjects\mmagic-main\data\Misc\misc\misc_gt.txt"  # 替换成输出的txt文件路径

    write_to_txt(image_folder_path, output_txt_file)
