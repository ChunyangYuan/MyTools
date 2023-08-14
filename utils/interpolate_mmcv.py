import os
import mmcv
from PIL import Image
from typing import Union, Tuple, Optional


def resize_image(image_folder_path: str,
                 output_folder: str,
                 scale_factor: Union[int, float, Tuple[int, int]] = 2,
                 interpolation: str = 'bicubic',
                 backend: Optional[str] = 'cv2') -> None:
    """
    resize_image resize image using interpolation methods.
    Interpolation method, accepted values are “nearest”, “bilinear”, “bicubic”, “area”, “lanczos” for 'cv2' backend, “nearest”, “bilinear” for 'pillow' backend.

    Args:
        image_folder_path (str): image folder path to be resized.
        output_folder (str): resized images save folder path.
        scale_factor (Union[int, float, Tuple[int, int]], optional): scale factor. Defaults to 2.
        interpolation (str, optional): interpolation method. Defaults to 'bicubic'.
        backend (Optional[str], optional): The image resize backend type. Options are cv2, pillow, None. If backend is None, the global imread_backend specified by mmcv.use_backend() will be used. Defaults to 'cv2'.

    Returns:
        _type_: None
    """
    img_list = os.listdir(image_folder_path)
    for img_name in img_list:
        image_path = os.path.join(image_folder_path, img_name)

        # 读取图像
        img = mmcv.imread(image_path, flag='color')

        # 调用imrescale函数进行缩放
        resized_img = mmcv.image.imrescale(
            img, scale_factor, interpolation=interpolation, backend=backend)

        # 生成保存的文件路径
        output_path = os.path.join(output_folder, img_name)

        # 保存缩放后的图像
        mmcv.imwrite(resized_img, output_path)

    print(f"Resized image saved at: {output_folder}")


if __name__ == "__main__":

    # 用法示例
    image_folder_path = 'E:\dataset\SIRSTdevkit-master\PNGImages'
    output_folder = 'E:\dataset\SIRSTdevkit-master\PNGImages_2x_bicubic'
    scale_factor = 2
    # Interpolation method, accepted values are “nearest”, “bilinear”, “bicubic”, “area”, “lanczos” for 'cv2' backend, “nearest”, “bilinear” for 'pillow' backend.
    interpolation_method = 'bicubic'
    backend = 'cv2'
    resize_image(
        image_folder_path, output_folder, scale_factor, interpolation_method, backend)
