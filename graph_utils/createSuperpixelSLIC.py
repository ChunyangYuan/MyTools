import cv2
import numpy as np
# 其中各个参数意义如下：
#     image ：输入图像
#     algorithm：选择要使用的算法变体：SLIC、SLICO（默认）和MSLIC三种可选
#     region_size：平均超像素大小，默认10
#     ruler：超像素平滑度，默认10
img = cv2.imread("sar_reshape.png")
#初始化slic项，超像素平均尺寸20（默认为10），平滑因子20
slic = cv2.ximgproc.createSuperpixelSLIC(img,algorithm=101,region_size=50,ruler = 39.0) 
slic.iterate(20)     #迭代次数，越大效果越好
mask_slic = slic.getLabelContourMask() #获取Mask，超像素边缘Mask==1
label_slic = slic.getLabels()        #获取超像素标签
number_slic = slic.getNumberOfSuperpixels()  #获取超像素数目
mask_inv_slic = cv2.bitwise_not(mask_slic)  
img_slic = cv2.bitwise_and(img,img,mask =  mask_inv_slic) #在原图上绘制超像素边界
cv2.imshow("img_slic",img_slic)
cv2.waitKey(0)
cv2.destroyAllWindows()
