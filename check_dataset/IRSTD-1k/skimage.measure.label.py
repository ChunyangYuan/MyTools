import skimage.measure
import skimage.io
import numpy as np

from PIL import Image
import numpy as np

# # 创建一个空白的灰度图像
# width = 200
# height = 200
# image = np.zeros((height, width), dtype=np.uint8)

# # 绘制左上角的白色区域
# image[10:30, :20] = 255

# # 绘制右上角的白色区域
# image[:20, -30:-10] = 255

# # 将 NumPy 数组转换为 PIL 图像
# pil_image = Image.fromarray(image, mode='L')

# # 保存图像
# pil_image.save('gray_image.png')
# # 读取图像
path = r'data\results\XDU1_mask.png'
image = skimage.io.imread(path, as_gray=True)
print(np.unique(image))
# 连通区域标记
labeled_image = skimage.measure.label(image)
unique = np.unique(labeled_image)
print(unique)

# 计算连通区域属性
props = skimage.measure.regionprops(labeled_image)

# 打印连通区域属性
for region in props:
    print("Label:", region.label)
    print("Area:", region.area)
    print("Centroid:", region.centroid)
    print("Bounding Box:", region.bbox)  # (ymin, xmin, ymax, xmax)
    print("Perimeter:", region.perimeter)
    print("...")
