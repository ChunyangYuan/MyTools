from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def test():
    # 创建一个256x256的全零矩阵，数据类型为uint8
    class_image = np.zeros((256, 256), dtype=np.uint8)

    # 将左上角设置为0
    class_image[:128, :128] = 0

    # 将右上角设置为1
    class_image[:128, 128:] = 1

    # 将左下角设置为2
    class_image[128:, :128] = 2

    # 将右下角设置为3
    class_image[128:, 128:] = 3

    # 将PIL图像转换为NumPy数组
    class_array = np.array(class_image)

    # 定义类别及其对应的颜色
    class_colors = {
        0: (0, 0, 0),       # 类别0的颜色为黑色
        1: (255, 0, 0),     # 类别1的颜色为红色
        2: (0, 255, 0),     # 类别2的颜色为绿色
        3: (0, 0, 255)      # 类别3的颜色为蓝色
    }

    # 创建新的RGB图像数组
    rgb_array = np.zeros(
        (class_array.shape[0], class_array.shape[1], 3), dtype=np.uint8)

    # 将类别映射为对应的颜色
    for class_id, color in class_colors.items():
        rgb_array[class_array == class_id] = color

    # 创建PIL图像对象
    rgb_image = Image.fromarray(rgb_array)

    # 使用Matplotlib显示图像
    plt.imshow(rgb_array)
    plt.axis('off')  # 可选，去除坐标轴
    plt.show()
    # 保存图像到文件
    rgb_image.save('map.png')


def draw_sar_gt(gt_path: str = r'F:\Dataset\SAR\gt.png'):
    img = Image.open(gt_path)
    class_array = np.array(img)
    sar_class_colors = {
        0: (0, 0, 0),       # 类别0的颜色为黑色
        1: (0, 0, 255),     # 类别1的颜色为蓝
        2: (0, 255, 0),     # 类别2的颜色为绿色
        3: (255, 255, 0),      # 类别3的颜色为黄
        4: (255, 0, 0)  # 红
    }

    # 创建新的RGB图像数组
    rgb_array = np.zeros(
        (class_array.shape[0], class_array.shape[1], 3), dtype=np.uint8)

    # 将类别映射为对应的颜色
    for class_id, color in sar_class_colors.items():
        rgb_array[class_array == class_id] = color

    # 创建PIL图像对象
    rgb_image = Image.fromarray(rgb_array)

    # 使用Matplotlib显示图像
    plt.imshow(rgb_array)
    plt.axis('off')  # 可选，去除坐标轴
    plt.show()
    # 保存图像到文件
    rgb_image.save('sar_gt.png')

    pass


def draw_classification_map(pred, name: str = 'sar'):
    if name == 'sar':
        color_map = {
            0: (0, 0, 255),
            1: (0, 255, 0),
            2: (255, 255, 0),
            3: (255, 0, 0),
        }
    elif name == 'munich':
        color_map = {
            0: (222, 184, 135),
            1: (0, 100, 0),
            2: (203, 0, 0),
            3: (0, 0, 100),
        }
    pred = np.array(pred).astype(np.uint8)
    # 创建新的RGB图像数组
    rgb_array = np.zeros(
        (class_array.shape[0], class_array.shape[1], 3), dtype=np.uint8)
    # 将类别映射为对应的颜色
    for class_id, color in color_map.items():
        rgb_array[pred == class_id] = color
    # 创建PIL图像对象
    rgb_image = Image.fromarray(rgb_array)
    # # 使用Matplotlib显示图像
    # plt.imshow(rgb_array)
    # plt.axis('off')  # 可选，去除坐标轴
    # plt.show()
    # 保存图像到文件
    rgb_image.save('munich_s1.png')


def draw_munich_s1_map(
        gt_path: str = r'F:\Dataset\multi_sensor_landcover_classification\annotations\munich_anno.tif'):
    img = Image.open(gt_path)
    class_array = np.array(img)
    print("sum(class=0)={}".format(np.sum(class_array == 0)))
    munich_class_colors = {
        0: (0, 0, 0),       # 类别0的颜色为黑色
        1: (222, 184, 135),     # 浅棕色，agriculture field
        2: (0, 100, 0),     # 类别2的颜色为深绿， forest
        3: (203, 0, 0),      # 深红，built-up
        4: (0, 0, 100)  # 深蓝，water body
    }

    # 创建新的RGB图像数组
    rgb_array = np.zeros(
        (class_array.shape[0], class_array.shape[1], 3), dtype=np.uint8)

    # 将类别映射为对应的颜色
    for class_id, color in munich_class_colors.items():
        rgb_array[class_array == class_id] = color

    # 创建PIL图像对象
    rgb_image = Image.fromarray(rgb_array)

    # 使用Matplotlib显示图像
    plt.imshow(rgb_array)
    plt.axis('off')  # 可选，去除坐标轴
    plt.show()
    # 保存图像到文件
    rgb_image.save('munich_s1.png')

    pass


# draw_sar_gt()
draw_munich_s1_map()
