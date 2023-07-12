import torch
import torch.nn.functional as F
import cv2
import os
import os.path as osp

# INTER_NEAREST 0- 最近邻插值法
# INTER_LINEAR 1 - 双线性插值法（默认）
# INTER_AREA 3- 基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
# INTER_CUBIC 2- 基于4x4像素邻域的3次插值法
# INTER_LANCZOS4 4- 基于8x8像素邻域的Lanczos插值


def img_interpolate_cv2(input_folder: str, output_folder: str, scale: int):
    if not osp.exists(input_folder):
        print('input_folder do not exist! please check out!')
        sys.exit(1)

    interpolation_mode = cv2.INTER_LANCZOS4
    output_folder += '_'+str(interpolation_mode)
    if not osp.exists(output_folder):
        os.mkdir(output_folder)

    # img_list = list(mmengine.scandir(input_folder))
    img_list = os.listdir(input_folder)
    img_list = [osp.join(input_folder, v) for v in img_list]

    for img_path in img_list:
        # flags：读取图片的方式，可选项
        # cv2.IMREAD_COLOR(1)：始终将图像转换为 3 通道BGR彩色图像，默认方式
        # cv2.IMREAD_GRAYSCALE(0)：始终将图像转换为单通道灰度图像
        # cv2.IMREAD_UNCHANGED(-1)：按原样返回加载的图像（使用Alpha通道）
        # cv2.IMREAD_ANYDEPTH(2)：在输入具有相应深度时返回16位/ 32位图像，否则将其转换为8位
        # cv2.IMREAD_ANYCOLOR(4)：以任何可能的颜色格式读取图像
        data = cv2.imread(img_path, flags=-1)
        # data = cv2.resize(data,dsize=(width,height),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(data, dsize=None,
                         fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(osp.join(output_folder, osp.basename(img_path)), img)
    else:
        print('interpolate over!')


input_path = r'F:\dataset\SIRSTdevkit-master\test_up'
scale = 2
output_path = input_path+'_scale'+str(scale)
img_interpolate_cv2(input_path, output_path, scale)
