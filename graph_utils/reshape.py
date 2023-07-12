import numpy as np
from PIL import Image
from PIL import ImageFilter


def reshape_img(img_path: str, save_path: str):
    image = Image.open(img_path)
    image_reshape = image.resize((360, 360))
    image_reshape.save(save_path)
    print('over!')


def filter_img(img_path: str, save_path: str):
    image = Image.open(image_path)
    image_filter = image.filter(ImageFilter.BoxBlur(7))
    for i in range(3):
        image_filter = image_filter.filter(ImageFilter.BLUR)
    image_filter.save(save_path)
    print('over!')
    pass


if __name__ == "__main__":
    pass
    image_path = r'F:\dataset\SAR\gt.png'
    save_path = r'F:\dataset\SAR\gt_reshape.png'
    reshape_img(image_path, save_path)
    # filter_img(image_path, save_path)
