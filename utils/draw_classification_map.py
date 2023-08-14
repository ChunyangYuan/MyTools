from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import mmcv


def draw_classification_map(pred, img_path: str, dataset_name: str = 'sar'):
    if dataset_name == 'sar':
        color_map = {
            0: (0, 0, 255),
            1: (0, 255, 0),
            2: (255, 255, 0),
            3: (255, 0, 0),
        }
    elif dataset_name == 'munich':
        color_map = {
            0: (222, 184, 135),
            1: (0, 100, 0),
            2: (203, 0, 0),
            3: (0, 0, 100),
        }
    elif dataset_name == 'air':
        color_map = {
            0: (0, 0, 0),
            1: (0, 0, 255),
            2: (0, 255, 0),
            3: (255, 0, 0),
            4: (0, 255, 255),
            5: (255, 255, 255),
            6: (255, 255, 0),
        }
    pred = np.array(pred).astype(np.uint8)
    # 创建新的RGB图像数组
    rgb_array = np.zeros(
        (pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
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
    rgb_image.save(img_path)


def draw(gt_path, out_dir, dataset_name='air'):
    img_list = os.listdir(gt_path)
    for img_name in img_list:
        # img = Image.open(os.path.join(gt_path, img_name))
        img = mmcv.imread(os.path.join(gt_path, img_name), flag='unchanged')
        img_path = os.path.join(out_dir, img_name)
        draw_classification_map(img, img_path, dataset_name)


if __name__ == "__main__":
    gt_path = r'F:\Dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\air_select\gt_'
    out_dir = r'F:\Dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\air_select'
    draw(gt_path, out_dir, 'air')
