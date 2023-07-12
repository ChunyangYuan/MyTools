import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

# for i in range(1, 25):
#     print('\'{}\': {},'.format(7, 100+i))
# print('\'{}\': {},'.format(7, 100+i))


name = r'24_50_0.6_20_segmentmap.npy'
segment = np.load(name)
# 24 -> 13
mapping = {'0': 101,
           '1': 101,
           '5': 101,
           '2': 102,
           '3': 103,
           '4': 103,
           '6': 103,
           '7': 104,
           '8': 105,
           '14': 105,
           '9': 106,
           '10': 106,
           '11': 107,
           '12': 108,
           '13': 108,
           '15': 109,
           '16': 110,
           '17': 110,
           '18': 111,
           '19': 111,
           '20': 112,
           '21': 112,
           '22': 113,
           '23': 113,


           }
# changing to 24-pro
mapping_changing = {'0': 101,
                    '7': 102,
                    '5': 103,
                    '1': 104,
                    '12': 105,
                    '13': 106,
                    '8': 107,
                    '2': 108,
                    '18': 109,
                    '19': 110,
                    '14': 111,
                    '15': 112,
                    '9': 113,
                    '10': 114,
                    '6': 115,
                    '3': 116,
                    '20': 117,
                    '21': 118,
                    '16': 119,
                    '17': 120,
                    '11': 121,
                    '4': 122,
                    '22': 123,
                    '23': 124,
                    }
print(len(set([i for i in mapping_changing.keys()])))
print(len(np.unique(segment)))
for i in np.unique(segment):
    segment[segment == i] = mapping_changing[str(i)]

segment -= 100
# np.save('13_50_0.6_20_segmentmap.npy', segment)
# np.save(r'./segments/24_pro_segmentmap.npy', segment)
image = io.imread(r'sar_reshape.png')
fig, ax = plt.subplots()
image = img_as_float(image)
ax.imshow(mark_boundaries(image, segment, color=[255, 255, 0]))
plt.axis("off")
# plt.savefig('13-segment.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

for i in np.unique(segment):
    mask = np.zeros_like(segment, dtype=np.uint8)
    mask[segment == i] = 255
    # plt.show(mask)
    cv2.imshow('mask'+str(i), mask)
    cv2.waitKey(0)
