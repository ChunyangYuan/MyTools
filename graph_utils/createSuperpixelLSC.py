import cv2
import numpy as np
# 其中各个参数意义如下：
#     image：输入图像
#     region_size ：平均超像素大小，默认10
#     ratio：超像素紧凑度因子，默认0.075

region_size = 50
ratio = 0.6
iterations = 20
img = cv2.imread("sar_reshape.png")
# img = cv2.imread("ppt-big-pixel.png")
lsc = cv2.ximgproc.createSuperpixelLSC(img,region_size =region_size,ratio=ratio)
lsc.iterate(20)
mask_lsc = lsc.getLabelContourMask() # 轮廓

label_lsc = lsc.getLabels() # segmentmap
number_lsc = lsc.getNumberOfSuperpixels()
print(number_lsc)
mask_inv_lsc = cv2.bitwise_not(mask_lsc)
img_lsc = cv2.bitwise_and(img, img, mask=mask_inv_lsc)


# (number_lsc + region_size + ratio + iterations).png
# image_name = str(number_lsc)+'_'+str(region_size) +'_'+ str(ratio) + '_'+str(iterations)+'.png'
# cv2.imwrite(image_name,img_lsc)
# mask_name = str(number_lsc)+'_'+str(region_size) + '_'+str(ratio)+ '_'+str(iterations) +'_contour'
# label_name = str(number_lsc)+'_'+str(region_size) + '_'+str(ratio)+ '_'+str(iterations) +'_segmentmap'
# np.save(mask_name, mask_lsc)
# np.save(label_name,label_lsc)


cv2.imshow("img_lsc", img_lsc)
cv2.waitKey(0)
cv2.destroyAllWindows()
