import numpy as np


def convert_seg_map_to_bboxes(seg_map):
    # 寻找每个目标的边界框坐标
    labels = np.unique(seg_map)
    bboxes = []
    for label in labels:
        if label == 0:  # 背景类别通常为0，忽略
            continue
        mask = (seg_map == label)
        indices = np.where(mask)
        xmin = np.min(indices[1])
        xmax = np.max(indices[1])
        ymin = np.min(indices[0])
        ymax = np.max(indices[0])
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes


# 示例使用
gt_seg_maps = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 0, 0, 2, 0],
                       [0, 0, 0, 2, 0]])

gt_bboxes = convert_seg_map_to_bboxes(gt_seg_maps)
print(gt_bboxes)
