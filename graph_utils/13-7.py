import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

for i in range(1, 14):
    print('\'{}\': {},'.format(7, 100+i))
    # print('\'{}\': {},'.format(7, 100+i))


name = r'13_50_0.6_20_segmentmap.npy'
segment = np.load(name)
# 24 -> 13
mapping = {'0': 101,
           '1': 101,
           '2': 102,
           '3': 103,
           '7': 103,
           '4': 104,
           '5': 104,
           '8': 104,
           '6': 105,
           '9': 105,
           '10': 106,
           '11': 107,
           '12': 107,


           }
# changing to 13-pro
mapping_changing = {'0': 101,
                    '3': 102,
                    '4': 103,
                    '1': 104,
                    '7': 105,
                    '5': 106,
                    '2': 107,
                    '10': 108,
                    '11': 109,
                    '8': 110,
                    '6': 111,
                    '9': 112,
                    '12': 113,
                    }
print(len(set([i for i in mapping_changing.keys()])))
print(len(np.unique(segment)))

for i in np.unique(segment):
    segment[segment == i] = mapping_changing[str(i)]

segment -= 100
# np.save('7_50_0.6_20_segmentmap.npy', segment)
np.save(r'./segments/13_pro_segmentmap.npy', segment)
image = io.imread(r'sar_reshape.png')
fig, ax = plt.subplots()
image = img_as_float(image)
ax.imshow(mark_boundaries(image, segment, color=[255, 255, 0]))
plt.axis("off")
# plt.savefig('7-segment.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

for i in np.unique(segment):
    mask = np.zeros_like(segment, dtype=np.uint8)
    mask[segment == i] = 255
    # plt.show(mask)
    cv2.imshow('mask'+str(i), mask)
    cv2.waitKey(0)
