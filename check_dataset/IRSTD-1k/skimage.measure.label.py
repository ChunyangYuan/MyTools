import skimage.measure
import skimage.io

# 读取图像
image = skimage.io.imread('your_image.png', as_gray=True)

# 连通区域标记
labeled_image = skimage.measure.label(image)

# 打印标记后的图像
print(labeled_image)
