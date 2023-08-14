import cv2
import os


def count_channels(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(image.shape) < 3:
        return 1  # 单通道图像
    else:
        return image.shape[-1]  # 多通道图像的通道数


def count_channel_types(image_folder):
    channel_counts = {}
    total_images = 0

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        if os.path.isfile(image_path):
            total_images += 1
            channels = count_channels(image_path)
            if channels in channel_counts:
                channel_counts[channels] += 1
            else:
                channel_counts[channels] = 1

    return channel_counts, total_images


image_folder = r'E:\dataset\SIRSTdevkit-master\PNGImages'

channel_counts, total_images = count_channel_types(image_folder)

print(f'Total images: {total_images}')

for channels, count in channel_counts.items():
    print(f'Images with {channels} channel(s): {count}')
