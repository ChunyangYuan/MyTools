import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 坐标信息
xmin, ymin = 0, 10
xmax, ymax = 21, 31

# 加载灰度图像
image_path = 'gray_image.png'
image = Image.open(image_path).convert('L')

# 创建绘图对象
fig, ax = plt.subplots()

# 显示灰度图像
ax.imshow(image, cmap='gray')

# 创建矩形框
rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')

# 将矩形框添加到图像上
ax.add_patch(rect)

# 显示图像
plt.show()
