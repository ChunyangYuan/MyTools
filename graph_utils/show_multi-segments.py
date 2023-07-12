# coding=utf-8
# 导入相应的python包
import argparse
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import scipy.io as sio


mat = sio.loadmat(r'C:\Users\LwhYcy\Desktop\U\segments\segments_49_24_13_7.mat')
seg1 = mat['segmentmaps'][0]
seg2 =mat['segmentmaps'][1] 
seg3  = mat['segmentmaps'][2]
seg4 = mat['segmentmaps'][3]
image = io.imread(r'C:\Users\LwhYcy\Desktop\U\slic\sar_reshape.png')
image = img_as_float(image)

segments = slic(image, n_segments=100, sigma=5)
fig = plt.figure("Superpixels -- %d segments" % (400))
plt.subplot(141)
# plt.title('image')
plt.imshow(mark_boundaries(image, seg1,color=[255,0,0]))
plt.subplot(142)
# plt.title('segments')
plt.imshow(mark_boundaries(image, seg2,color=[255,0,0]))
plt.subplot(143)
# plt.title('image and segments')
# mode
# string in {‘thick’, ‘inner’, ‘outer’, ‘subpixel’}, optional
plt.imshow(mark_boundaries(image, seg3,color=[255,0,0]))
plt.subplot(144)
plt.imshow(mark_boundaries(image, seg4,color=[255,0,0]))
plt.show()

