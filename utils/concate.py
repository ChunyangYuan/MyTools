import numpy as np
from PIL import Image
import os
import re
import scipy.io as sio


def concate(src, save):
    img_lst = os.listdir(src)
    lst_np = np.array(img_lst)
    lst_np = lst_np.reshape(-1,16)
    # ptn = re.compile("[0-9]+")
    # name = "AIR-PolarSAR-Seg-1_HH_1.tiff"
    for group in lst_np:
        # No_ = group[0].findall(ptn)
        for i in range(4):
            HH_name = group[i]
            tmp = list(HH_name)
            tmp[-8] = 'V'
            HV_name = "".join(tmp)
            tmp = list(HH_name)
            tmp[-9] = 'V'
            VH_name = "".join(tmp)
            tmp[-8] = 'V'
            VV_name = "".join(tmp)
            HH = Image.open(os.path.join(src, HH_name))
            HH_np = np.array(HH).reshape(-1)
            HV = Image.open(os.path.join(src, HV_name))
            HV_np = np.array(HV).reshape(-1)
            VH = Image.open(os.path.join(src, VH_name))
            VH_np = np.array(VH).reshape(-1)
            VV = Image.open(os.path.join(src, VV_name))
            VV_np = np.array(VV).reshape(-1)
            mat = np.zeros(HH_np.size*4).reshape(-1,4)
            for j in range(HH_np.size):
                mat[j][0] = HH_np[j]
                mat[j][1] = HV_np[j]
                mat[j][2] = VH_np[j]
                mat[j][3] = VV_np[j]
            mat = mat.reshape(HH.width, HH.height, 4)
            mat_name = HH_name[:-10] + '_' + str(i+1) + ".mat"
            mat_name = os.path.join(save, mat_name)
            sio.savemat(mat_name, {'data':mat})
    else:
        print('concate over!')
    pass


if __name__ == "__main__":
    path = r"F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\images"
    # path = r"F:\dataset\Raw_AIR-PolarSAR-Seg\abc"
    save_path =  r"F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\mat"
    concate(path, save_path)
    pass