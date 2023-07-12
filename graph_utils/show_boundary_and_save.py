import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io


name = r'49_50_0.6_20_segmentmap.npy'
segment = np.load(name)

image = io.imread(r'sar_reshape.png')
fig, ax = plt.subplots()
image = img_as_float(image)
ax.imshow(mark_boundaries(image, segment, color=[255, 255, 0]))
plt.axis("off")
plt.savefig('49-segment.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

for i in np.unique(segment):
    mask = np.zeros_like(segment, dtype=np.uint8)
    mask[segment == i] = 255
    # plt.show(mask)
    cv2.imshow('mask'+str(i), mask)
    cv2.waitKey(0)
