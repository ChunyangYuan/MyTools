from PIL import Image
import matplotlib.pyplot as plt


def png_to_eps(input_file, output_file):
    # 打开PNG图片
    img = Image.open(input_file)

    # 获取图片宽度和高度
    width, height = img.size

    # 创建一个新的画布
    fig, ax = plt.subplots(
        figsize=(width/100, height/100))  # 设置画布大小，需要根据图片大小调整

    # 绘制图片到画布上
    ax.imshow(img)

    # 去除坐标轴
    ax.axis('off')

    # 保存画布为EPS格式
    plt.savefig(output_file, format='eps', dpi=1000, bbox_inches='tight')

    # 关闭画布
    plt.close()


if __name__ == "__main__":
    # 替换为你的输入PNG图片文件名
    input_png_file = r"C:\Users\LwhYcy\Desktop\classification_map\sar\map\color_gt.png"
    output_eps_file = r"color_gt.eps"  # 替换为你的输出EPS图片文件名
    png_to_eps(input_png_file, output_eps_file)
