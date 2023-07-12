import numpy as np
from PIL import Image
import os


def create_gt(path, save, mapping):
    if not os.path.exists(save):
        os.mkdir(save)

    img = Image.open(path)
    img_np = np.array(img)
    img_np = img_np.reshape(-1, 3)
    gt = np.ones((img_np.shape[0]))
    for i, pixel in enumerate(img_np):
        gt[i] = mapping.get(tuple(pixel))
    gt = gt.reshape(img.width,-1)
    gt = gt.astype(np.int8)
    gt = Image.fromarray(gt)
    name = os.path.basename(path)
    name = name.split('.')[0]+'_fined'+'.'+name.split('.')[-1]
    gt.save(os.path.join(save,name))
    print('create_gt over!')


if __name__ == "__main__":
    src = r"F:\dataset\SAR\sar_gt.png"
    dst = r'F:\dataset\SAR'
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
    mapping_sar = {
        (0,0,0):0, # 当做背景, 不统计训练数据
        (0,0,255):1,
        (0,255,0):2,
        (255,255,0):3,
        (255,0,0):4,
    }
    # mapping_AIR_PolSAR_Seg = {
    #     (0,0,0):0, # 当做背景, 不统计训练数据
    #     (0,0,255):1,
    #     (0,255,0):2,
    #     (255,0,0):3,
    #     (0,255,255):4,
    #     (255,255,255):5,
    #     (255,255,0):6
    # }
    create_gt(src,dst, mapping_sar)