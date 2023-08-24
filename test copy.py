
import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_gaussian_matrix(rows, cols):
    center_row = rows // 2
    center_col = cols // 2

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    pos = np.dstack((x, y))
    center = np.array([center_col, center_row])

    covariance = np.eye(2) * (0.1 * max(rows, cols)) ** 2
    gauss_matrix = multivariate_normal.pdf(pos, mean=center, cov=covariance)
    # Normalize the Gaussian matrix
    normalized_gauss_matrix = gauss_matrix / np.sum(gauss_matrix)
    return normalized_gauss_matrix


# 定义图像大小和均值
image_size = 512

# 定义最大偏置距离
offset = 20
object_max_size = 7
# 随机生成初始物体的宽度、高度和左上角坐标
init_width = np.random.randint(5, 20)
init_height = np.random.randint(5, 20)
init_x = np.random.randint(0, image_size)
init_y = np.random.randint(0, image_size)

# x_center = init_x + init_width // 2
# x_center = init_y + init_height // 2


floating_window_x1 = max(0, init_x - offset)
floating_window_y1 = max(0, init_y - offset)
floating_window_x2 = min(image_size, init_x + init_width + offset)
floating_window_y2 = min(image_size, init_y + init_height + offset)
floating_window_width = floating_window_x2 - floating_window_x1
floating_window_height = floating_window_y2 - floating_window_y1


normalized_gauss_matrix = generate_gaussian_matrix(
    floating_window_height, floating_window_width)

plt.imshow(normalized_gauss_matrix, cmap='viridis', origin='upper')
plt.colorbar()
plt.title('2D Gaussian Distribution Matrix')
plt.show()
# 在mask图像上绘制初始物体
mask = np.zeros((image_size, image_size), dtype=np.uint8)
mask[init_y:init_y+init_height, init_x:init_x+init_width] = 255

# # 定义协方差矩阵
# covariance_matrix = np.array([[1, 0], [0, 1]])

# # 生成二维高斯分布的概率密度函数
# # x = np.linspace(0, image_size - 1, image_size)
# # y = np.linspace(0, image_size - 1, image_size)
# # X, Y = np.meshgrid(x, y)
# # pos = np.dstack((X, Y))
# length = 1
# x = np.linspace(max(0, init_x - offset), min(image_size,
#                 init_y + init_height + offset), image_size)
# y = np.linspace(0, image_size - 1, image_size)
# X, Y = np.meshgrid(x, y)
# pos = np.dstack((X, Y))
# rv = multivariate_normal([x_center, x_center], covariance_matrix)

# # 计算概率密度函数值
# pdf_values = rv.pdf(pos)

# # 归一化概率密度函数值
# normalized_pdf_values = pdf_values / np.sum(pdf_values)

# 采样随机坐标位置和目标大小
num_objects = 5
cnt = 0
sampled_objects = []
# 生成副本的数量
for _ in range(num_objects):
    while True:
        # 使用概率密度函数值进行不重复加权采样
        idx = np.random.choice(floating_window_width * floating_window_height, replace=False,
                               p=normalized_gauss_matrix.flatten())
        # 将索引转换为左上角坐标
        y = idx // floating_window_height + floating_window_y1
        x = idx % floating_window_width + floating_window_x1
        print('x={},y={}'.format(x, y))

        # 随机生成目标的宽度和高度，不超过7像素
        width = np.random.randint(1, object_max_size+1)
        height = np.random.randint(1, object_max_size+1)
        if x + width > image_size or y + height > image_size:
            continue
        # 确保物体不超出图像边界
        xmin = max(x, 0)
        ymin = max(y, 0)
        xmax = min(x + width, image_size)
        ymax = min(y + height, image_size)

        # 检查物体是否与已有物体重叠，此外控制所有目标相邻距离>=margin
        margin = 2
        overlap = np.any(mask[max(ymin-margin, 0):min(ymax+margin, image_size),
                         max(xmin-margin, 0):min(xmax+margin, image_size)] == 255)

        if not overlap:
            cnt += 1
            mask[ymin:ymax, xmin:xmax] = 255
            sampled_objects.append((xmin, ymin, xmax, ymax))
            break

# # 在mask图像上绘制副本
for xmin, ymin, xmax, ymax in sampled_objects:
    print('{},{},{},{}'.format(xmin, ymin, xmax, ymax))
print(cnt)
# 显示mask图像
cv2.imshow('Mask Image', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
if mode == 'localresize':
    for contour in contours:
        # Calculate the bounding rectangle for the contour
        original_object_x1, original_object_y1, original_object_w, original_object_h = cv2.boundingRect(
            contour)

        if original_object_w >= object_max_size or original_object_h >= object_max_size:

            # Calculate the scaling factor to shrink to target size
            scale_factor = min(
                object_max_size / original_object_w, object_max_size / original_object_h)

            # Calculate the new width and height
            new_w = int(original_object_w * scale_factor)
            new_h = int(original_object_h * scale_factor)

            # Calculate the new top-left corner
            new_x = int(original_object_x1 +
                        (original_object_w - new_w) / 2)
            new_y = int(original_object_y1 +
                        (original_object_h - new_h) / 2)

            # Crop the original region of interest and enlarge roi
            padding = 1
            img_height, img_width = image.shape[:2]

            roi_image = image[max(
                0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)]
            roi_mask = mask[max(
                0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)]

            # Resize the cropped region to target size
            resized_roi_image = cv2.resize(
                roi_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # roi_image2 = cv2.resize(
            #     resized_roi_image, (original_object_w+2*padding, original_object_h+2*padding), interpolation=cv2.INTER_LINEAR)
            # 定义腐蚀操作的结构元素（内核）
            # kernel_size = (7, 7)  # 调整内核的大小
            # # 调整内核的形状,也可以使用 cv2.MORPH_ELLIPSE 或 cv2.MORPH_CROSS cv2.MORPH_RECT
            # kernel_shape = cv2.MORPH_RECT

            # kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
            kernel = np.ones((5, 5), np.uint8)  # 5x5的正方形内核,可以根据需要调整大小

            # 进行腐蚀操作
            # eroded_image = cv2.erode(roi_image, kernel, iterations=1)
            kernel_size = (9, 9)
            # blurred = cv2.blur(roi_image, kernel_size)
            blurred = cv2.medianBlur(roi_image, ksize=11)
            # blurred = cv2.GaussianBlur(roi_image, kernel_size, 0)
            eroded_image = cv2.erode(blurred, kernel, iterations=1)
            # blurred_mask = cv2.GaussianBlur(roi_mask, kernel_size, 0)
            # blurred_mask[blurred_mask != 0] = 255

            # # blank_image = np.zeros_like(image)
            # result = cv2.bitwise_or(roi_image, roi_image2)
            resized_roi_mask = cv2.resize(
                roi_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_roi_mask[resized_roi_mask != 0] = 255

            # Fill the surrounding region in image with average pixel value of the eight surrounding boxes
            # avg_pixel_value = calculate_mean_around_box(
            #     image, original_object_x1, original_object_y1, original_object_w, original_object_h)
            # image[max(
            #     0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)] = avg_pixel_value
            # Update image with resized region
            image[max(
                0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)] = eroded_image
            # image[new_y:new_y + new_h, new_x:new_x + new_w] = blurred
            image[new_y:new_y + new_h, new_x:new_x +
                  new_w] = resized_roi_image
            # blurred = cv2.GaussianBlur(roi_image, kernel_size, 0)

            # Update mask with resized region and fill surrounding with 0
            mask[original_object_y1:original_object_y1+original_object_h,
                 original_object_x1:original_object_x1+original_object_w] = 0
            mask[new_y:new_y + new_h, new_x:new_x + new_w] = resized_roi_mask
    # image[new_y:new_y + new_h, new_x:new_x + new_w] = blurred
    image[new_y:new_y + new_h, new_x:new_x +
          new_w] = resized_roi_image
    # blurred = cv2.GaussianBlur(roi_image, kernel_size, 0)

    # Update mask with resized region and fill surrounding with 0
    mask[y:y+h, x:x+w] = 0
    mask[new_y:new_y + new_h, new_x:new_x + new_w] = resized_roi_mask
    # mask[max(
