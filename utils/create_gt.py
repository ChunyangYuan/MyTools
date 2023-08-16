import numpy as np
from PIL import Image
import os


def create_gt(img_path, save_path, mapping):

    img = Image.open(img_path)
    img_np = np.array(img)
    img_np = img_np.reshape(-1, 3)
    gt = np.ones((img_np.shape[0]))
    for i, pixel in enumerate(img_np):
        gt[i] = mapping.get(tuple(pixel))
    gt = gt.reshape(img.width, -1)
    gt = gt.astype(np.int8)
    gt = Image.fromarray(gt)

    gt.save(save_path)
    print('{} done!'.format(os.path.basename(save_path)))


if __name__ == "__main__":

    """
    color_map:
        categories      color(RGB)               class_id
        ------------------------------------------------------
        Unknown         black(0,0,0)                0
        Water           blue(0,0,255)               1
        Forest          green(0,255,0)              2
        Farmland        yellow(255,255,0)           3
        Building        red(255,0,0)                4
    """
    color_map = {
        (0, 0, 0): 0,  # 当做背景, 不统计训练数据
        (0, 0, 255): 1,
        (0, 255, 0): 2,
        (255, 255, 0): 3,
        (255, 0, 0): 4,
    }

    src = r"F:\Dataset\SAR\color_gt.png"
    dst = r'gt.png'
    create_gt(src, dst, color_map)
