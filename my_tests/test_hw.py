import cv2

# 以灰度模式读取图像
image_gray = cv2.imread(
    r'E:\dataset\SIRSTdevkit-master\PNGImages\211_HD_1101.png', cv2.IMREAD_GRAYSCALE)

# 获取图像的高度和宽度
height, width = image_gray.shape

print("图像的高度：", height)
print("图像的宽度：", width)
