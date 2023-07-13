from PIL import Image
import numpy as np
import re
import os


def classify(src:str, save:str, pattern):
    assert os.path.exists(src)
    if not os.path.exists(save):
        os.mkdir(save)

    img_lst = os.listdir(src)
    for img_name in img_lst:
        if pattern.match(img_name):
            img_path = os.path.join(src,img_name)
            img = Image.open(img_path)
            name = os.path.join(save, img_name)
            img.save(name)
    else:
        print('classify over!')
    pass



if __name__ == "__main__":
    ptn = re.compile(".*gt.png$")
    train_path = r'F:\dataset\Raw_AIR-PolarSAR-Seg\train_set'
    test_path = r"F:\dataset\Raw_AIR-PolarSAR-Seg\test_changed"
    save_dir = r"F:\dataset\Raw_AIR-PolarSAR-Seg\dataset\mask"
    # classify(train_path, save_dir, ptn)
    classify(test_path, save_dir, ptn)
    pass