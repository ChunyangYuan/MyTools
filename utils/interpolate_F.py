import torch
import torch.nn.functional as F
# import cv2
import os
import os.path as osp
import numpy as np
import PIL.Image as Image


def img_interpolate_F(input_folder: str, output_folder: str, scale: int, mode: str) -> None:
    """
    img_interpolate_F 使用torch.nn.functional的插值函数interpolate生成图片并保存

    Args:
        input_folder (str): 输入图片路径
        output_folder (str): 保存路径
        scale (int): 缩放因子
        mode (str): 插值方法
    """

    if not osp.exists(input_folder):
        print('input_folder do not exist! please check out!')
        sys.exit(1)

    if not osp.exists(output_folder):
        os.mkdir(output_folder)

    # img_list = list(mmengine.scandir(input_folder))
    img_list = os.listdir(input_folder)
    img_list = [osp.join(input_folder, v) for v in img_list]

    for img_path in img_list:
        img = np.array(Image.open(img_path))
        img = torch.from_numpy(img)
        img = torch.permute(img, (2, 0, 1))
        if mode == 'bilinear' or mode == 'bicubic':
            scaled_img = F.interpolate(torch.unsqueeze(
                img, 0), scale_factor=scale, mode=mode, antialias=True)
        else:
            scaled_img = F.interpolate(torch.unsqueeze(
                img, 0), scale_factor=scale, mode=mode, antialias=False)
        img = scaled_img.squeeze()
        img = torch.permute(img, (1, 2, 0))
        img = img.numpy()
        img = Image.fromarray(img)
        img.save(osp.join(output_folder, osp.basename(img_path)))
    else:
        print('interpolate over!')


if __name__ == "__main__":
    input_path = r'F:\dataset\SIRSTdevkit-master\mini_Misc'
    scale = 4
    # mode = 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'nearest'
    mode = 'bicubic'
    output_path = input_path+'_x'+str(scale)+'_'+mode
    img_interpolate_F(input_path, output_path, scale, mode)
