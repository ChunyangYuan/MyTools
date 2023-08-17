# Example long lists for x, y, w, h
all_xmin = [10, 20, 30, 40, 50]
all_ymin = [20, 30, 40, 50, 60]
all_xmax = [60, 70, 80, 90, 100]
all_ymax = [50, 60, 70, 80, 90]
single_object_bboxs = []


# Create single_object_bbox using list comprehension
single_object_bbox = [(xmin, ymin, xmax, ymax)
                      for xmin, ymin, xmax, ymax in zip(all_xmin, all_ymin, all_xmax, all_ymax)]


print("Single object bounding boxes:", single_object_bbox)


single_object_bboxs.extend(single_object_bbox)
single_object_bboxs.extend(single_object_bbox)
print(single_object_bboxs)
