import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

for i in range(1, 50):
    print('\'{}\': {},'.format(7, i))
#     # print('\'{}\': {},'.format(7, 100+i))


name = r'49_50_0.6_20_segmentmap.npy'
segment = np.load(name)
# merging
mapping_merging = {'0': 101,
                   '7': 101,
                   '1': 102,
                   '2': 102,
                   '3': 103,
                   '10': 103,
                   '4': 104,
                   '5': 104,
                   '6': 105,
                   '13': 105,
                   '8': 106,
                   '9': 106,
                   '11': 107,
                   '12': 107,
                   '14': 108,
                   '15': 108,
                   '16': 109,
                   '23': 109,
                   '17': 110,
                   '24': 110,
                   # ===============
                   '18': 111,
                   '25': 111,
                   '19': 112,
                   '20': 112,
                   '21': 113,
                   '28': 113,
                   '22': 114,
                   '29': 114,
                   '30': 115,
                   '37': 115,
                   '31': 116,
                   '32': 116,
                   '26': 117,
                   '33': 117,
                   '27': 118,
                   '34': 118,
                   '35': 119,
                   '42': 119,
                   '36': 120,
                   '43': 120,
                   '45': 121,
                   '44': 121,
                   '38': 121,
                   '39': 122,
                   '46': 122,
                   '40': 123,
                   '47': 123,
                   '41': 124,
                   '48': 124,
                   }
# change the id of nodes.
mapping_changing = {'0': 101,
                    '7': 102,
                    '8': 103,
                    '1': 104,
                    '14': 105,
                    '15': 106,
                    '9': 107,
                    '2': 108,
                    '21': 109,
                    '22': 110,
                    '23': 111,
                    '16': 112,
                    '10': 113,
                    '3': 114,
                    '28': 115,
                    '29': 116,
                    '30': 117,
                    '24': 118,
                    '17': 119,
                    '18': 120,
                    '11': 121,
                    '4': 122,
                    '35': 123,
                    '36': 124,
                    '37': 125,
                    '31': 126,
                    '32': 127,
                    '25': 128,
                    '26': 129,
                    '19': 130,
                    '12': 131,
                    '5': 132,
                    '42': 133,
                    '43': 134,
                    '44': 135,
                    '38': 136,
                    '39': 137,
                    '40': 138,
                    '33': 139,
                    '34': 140,
                    '27': 141,
                    '20': 142,
                    '13': 143,
                    '6': 144,
                    '45': 145,
                    '46': 146,
                    '47': 147,
                    '48': 148,
                    '41': 149,
                    }
print(len(set([i for i in mapping_changing.keys()])))
print(len(np.unique(segment)))
# 49 -> 24
for i in np.unique(segment):
    segment[segment == i] = mapping_changing[str(i)]

segment -= 100
# np.save('24_50_0.6_20_segmentmap.npy', segment)
# np.save(r'./segments/49_pro_segmentmap.npy', segment)
image = io.imread(r'sar_reshape.png')
fig, ax = plt.subplots()
image = img_as_float(image)
ax.imshow(mark_boundaries(image, segment, color=[255, 255, 0]))
plt.axis("off")
# plt.savefig('24-segment.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

print(len(np.unique(segment)))
for i in np.unique(segment):
    mask = np.zeros_like(segment, dtype=np.uint8)
    mask[segment == i] = 255
    # plt.show(mask)
    cv2.imshow('mask'+str(i), mask)
    cv2.waitKey(0)
