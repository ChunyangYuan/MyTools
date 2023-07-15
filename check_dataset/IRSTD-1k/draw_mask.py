from PIL import Image
import numpy as np

# 创建一个空白的灰度图像
width = 200
height = 200
image = np.zeros((height, width), dtype=np.uint8)

# 绘制左上角的白色区域
image[10:30, :20] = 255

# 绘制右上角的白色区域
image[:20, -30:-10] = 255

# 将 NumPy 数组转换为 PIL 图像
pil_image = Image.fromarray(image, mode='L')

# 保存图像
pil_image.save('gray_image.png')
