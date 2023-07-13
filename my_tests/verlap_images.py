import numpy as np
import matplotlib.pyplot as plt


def overlap_images(big_image, small_image, position):
    """
    在大图片上重叠放置小图片

    参数：
    big_image：大图片的 NumPy 数组
    small_image：小图片的 NumPy 数组
    position：小图片的放置位置，以左上角为原点的 (x, y) 坐标

    返回：
    result：放置了小图片的新图片的 NumPy 数组
    """

    # 获取小图片的尺寸
    small_height, small_width, _ = small_image.shape

    # 获取放置位置的坐标
    x, y = position

    # 确定小图片在大图片上的切片范围
    slice_x = slice(x, x + small_width)
    slice_y = slice(y, y + small_height)

    # 将小图片放置在大图片上
    result = np.copy(big_image)
    result[slice_y, slice_x] = small_image

    return result


# 创建大图片和小图片的示例数据
big_image = np.zeros((200, 200, 3), dtype=np.uint8)
small_image = np.ones((50, 50, 3), dtype=np.uint8) * 255

# 将小图片放置在大图片上
position = (75, 75)
result = overlap_images(big_image, small_image, position)

# 显示重叠后的图片
plt.imshow(result)
plt.axis('off')
plt.show()
