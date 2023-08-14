import numpy as np
from PIL import Image
import os


def create_gt(img_path, save_path, mapping):
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

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


def mk_gt(mask_path: str, out_dir: str, mapping: dict):
    mask_list = os.listdir(mask_path)
    for img_name in mask_list:
        img_path = os.path.join(mask_path, img_name)
        save_path = os.path.join(out_dir, img_name)
        create_gt(img_path, save_path, mapping)


if __name__ == "__main__":
    src = r"F:\Dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\air_select\mask"
    dst = r'F:\Dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\gt'
    """
    mapping_sar:
        categories      color(RGB)               class_id
        ------------------------------------------------------
        Unknown         black(0,0,0)                0
        Water           blue(0,0,255)               1
        Forest          green(0,255,0)              2
        Farmland        yellow(255,255,0)           3
        Building        red(255,0,0)                4

    mapping_AIR_PolSAR_Seg:
        categories          color(RGB)              class_id
        -------------------------------------------------------
        Background          black(0,0,0)                0
        Industrial          blue(0,0,255)               1
        Natural             green(0,255,0)              2
        Land Use            red(255,0,0)                3
        Water               cyan(0,255,255)             4
        Other               white(255,255,255)          5
        Housing             yellow(255,255,0)           6
    """
    # color_map = {
    #     (0,0,0):0, # 当做背景, 不统计训练数据
    #     (0,0,255):1,
    #     (0,255,0):2,
    #     (255,255,0):3,
    #     (255,0,0):4,
    # }
    color_map = {
        (0, 0, 0): 0,  # 当做背景, 不统计训练数据
        (0, 0, 255): 1,
        (0, 255, 0): 2,
        (255, 0, 0): 3,
        (0, 255, 255): 4,
        (255, 255, 255): 5,
        (255, 255, 0): 6
    }
    mk_gt(src, dst, color_map)
