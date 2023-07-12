import numpy as np
import scipy.io as sio


# seg7 = np.load(r'7_50_0.6_20_segmentmap.npy')
# seg13 = np.load(r'13_50_0.6_20_segmentmap.npy')
# seg24 = np.load(r'24_50_0.6_20_segmentmap.npy')
# seg49 = np.load(r'49_50_0.6_20_segmentmap.npy')


seg7 = np.load(r'./segments/7_pro_segmentmap.npy')
seg13 = np.load(r'./segments/13_pro_segmentmap.npy')
seg24 = np.load(r'./segments/24_pro_segmentmap.npy')
seg49 = np.load(r'./segments/49_pro_segmentmap.npy')


segments = [seg49, seg24, seg13, seg7]

sio.savemat('segments_49_24_13_7_pro.mat', {'segmentmaps': segments})

pass
