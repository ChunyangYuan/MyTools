import cv2
import numpy as np
import mmcv
import mmengine
import os.path as osp
import os


# def pick_3channel_img(input_folder: str, output_folder: str = r''):
#     if not osp.exists(input_folder):
#         print('input_folder do not exist! please check out!')
#         sys.exit(1)

#     # if not osp.exists(output_folder):
#     #     os.mkdir(output_folder)

#     img_list = list(mmengine.scandir(input_folder))
#     # img_list = os.listdir(input_folder)
#     img_list = [osp.join(input_folder, v) for v in img_list]
#     cnt = 0
#     for img_path in img_list:
#         img = mmcv.imread(img_path, flag='unchanged')
#         if len(img.shape) == 3 and img.shape[-1] == 3:
#             # if len(img.shape) == 2:
#             cnt += 1
#             # if len(img.shape) == 2:
#             # mmcv.imwrite(img, osp.join(output_folder, osp.basename(img_path)))
#     else:
#         print('pick over!cnt = {}'.format(cnt))


# data_path = r'E:\dataset\SIRSTdevkit-master\PNGImages_2X'
# # output = r'F:\dataset\SIRSTdevkit-master\Misc_ch3'
# pick_3channel_img(data_path)


def pick_3channel_img(input_folder: str, output_folder: str = ''):
    if not osp.exists(input_folder):
        print('input_folder does not exist! Please check.')
        return

    cnt = 0
    for filename in mmengine.scandir(input_folder):
        img_path = osp.join(input_folder, filename)

        # Load the image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Check if the image has 3 channels
        if img.shape[-1] == 3:
            cnt += 1

            # # Save the image to the output folder
            # if output_folder:
            #     output_path = osp.join(output_folder, filename)
            #     cv2.imwrite(output_path, img)

    print('Pick over! Count = {}'.format(cnt))


data_path = r'E:\dataset\SIRSTdevkit-master\PNGImages_2X'
# output = r'F:\dataset\SIRSTdevkit-master\Misc_ch3'
pick_3channel_img(data_path)
