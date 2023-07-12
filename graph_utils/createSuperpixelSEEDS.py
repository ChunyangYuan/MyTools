import cv2
import numpy as np
# 其中各个参数意义如下：
#     image_width ：输入图像宽度
#     image_height： 输入图像高度
#     image_channels ：输入图像通道数
#     num_superpixels ：期望超像素数目
#     num_levels ：块级别数，值越高，分段越准确，形状越平滑，但需要更多的内存和CPU时间。
#     histogram_bins： 直方图bins数，默认5
#     double_step： 如果为true，则每个块级别重复两次以提高准确性默认false。
img = cv2.imread("sar_reshape.png")
#初始化seeds项，注意图片长宽的顺序
seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1],img.shape[0],img.shape[2],50,30,3,5,True)
seeds.iterate(img,10)  #输入图像大小必须与初始化形状相同，迭代次数为10
mask_seeds = seeds.getLabelContourMask()
label_seeds = seeds.getLabels()
number_seeds = seeds.getNumberOfSuperpixels()
mask_inv_seeds = cv2.bitwise_not(mask_seeds)
img_seeds = cv2.bitwise_and(img,img,mask =  mask_inv_seeds)
cv2.imshow("img_seeds",img_seeds)
cv2.waitKey(0)
cv2.destroyAllWindows()
