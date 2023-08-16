import cv2
import numpy as np
import numpy
# import PIL.Image as Image
import os
import random
from typing import Tuple


def shrink_large_objects(image: np.ndarray,
                         mask: np.ndarray,
                         target_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    shrink_large_objects 缩小大目标

    Args:
        image (np.ndarray): 图像数据
        mask (np.ndarray): 掩码数据
        target_size (int, optional): 目标最大尺寸. Defaults to 7.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回处理后的图像和掩码
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        if w > target_size or h > target_size:
            # Calculate the scaling factor to shrink to target size
            scale_factor = min(target_size / w, target_size / h)

            # Calculate the new width and height
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)

            # Calculate the new top-left corner
            new_x = int(x + (w - new_w) / 2)
            new_y = int(y + (h - new_h) / 2)

            # Crop the original region of interest and enlarge roi
            padding = 5
            img_height, img_width = image.shape[:2]

            roi_image = image[max(
                0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)]
            roi_mask = mask[max(
                0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)]

            # Resize the cropped region to target size
            resized_roi_image = cv2.resize(
                roi_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # roi_image2 = cv2.resize(
            #     resized_roi_image, (w+2*padding, h+2*padding), interpolation=cv2.INTER_LINEAR)
            # 定义腐蚀操作的结构元素（内核）
            # kernel_size = (7, 7)  # 调整内核的大小
            # # 调整内核的形状，也可以使用 cv2.MORPH_ELLIPSE 或 cv2.MORPH_CROSS cv2.MORPH_RECT
            # kernel_shape = cv2.MORPH_RECT

            # kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
            kernel = np.ones((5, 5), np.uint8)  # 5x5的正方形内核，可以根据需要调整大小

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
            #     image, x, y, w, h)
            # image[max(
            #     0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)] = avg_pixel_value
            # Update image with resized region
            image[max(
                0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)] = eroded_image
            # image[new_y:new_y + new_h, new_x:new_x + new_w] = blurred
            image[new_y:new_y + new_h, new_x:new_x + new_w] = resized_roi_image
            # blurred = cv2.GaussianBlur(roi_image, kernel_size, 0)

            # Update mask with resized region and fill surrounding with 0
            mask[y:y+h, x:x+w] = 0
            mask[new_y:new_y + new_h, new_x:new_x + new_w] = resized_roi_mask
            # mask[max(
            #     0, y - padding):min(img_height, y + h + padding), max(0, x - padding):min(img_width, x + w + padding)] = blurred_mask
            # 增强图像目标
            # 定义亮度增加的值
            # brightness_increase = 50

            # # 使用布尔索引选择满足条件的像素位置
            # selected_pixels = mask == 255

            # # 在满足条件的位置上增加亮度，同时确保不超过255
            # # image[selected_pixels] = np.minimum(
            # #     image[selected_pixels] + brightness_increase, 230)
            # adjusted_pixels = image[selected_pixels] + brightness_increase
            # adjusted_pixels[adjusted_pixels > 255] = 255
            # image[selected_pixels] = adjusted_pixels

    return image, mask


def calculate_mean_around_box(image: np.ndarray,
                              x: int,
                              y: int,
                              width: int,
                              height: int) -> numpy.uint8:
    """
    calculate_mean_around_box 以给定坐标框为中心，计算其八邻域像素均值

    Args:
        image (np.ndarray): 图像数据
        x (int): 左上角x坐标
        y (int): 左上角y坐标
        width (int): 目标框宽度
        height (int): 目标框高度

    Returns:
        numpy.uint8: 返回八邻域像素均值
    """
    # Calculate the coordinates of the eight surrounding boxes
    boxes = [
        (x - width, y - height, x, y),             # 左上角
        (x, y - height, x + width, y),             # 上
        (x + width, y - height, x + width * 2, y),  # 右上角
        (x - width, y, x, y + height),             # 左
        (x + width, y, x + width * 2, y + height),  # 右
        (x - width, y + height, x, y + height * 2),  # 左下角
        (x, y + height, x + width, y + height * 2),  # 下
        (x + width, y + height, x + width * 2, y + height * 2)  # 右下角
    ]

    # Ensure box boundaries are within the image dimensions
    img_height, img_width = image.shape[:2]
    boxes = [(max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2))
             for x1, y1, x2, y2 in boxes]

    means = []
    for box in boxes:
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        mean = np.mean(roi)
        means.append(mean)

    overall_mean = np.mean(means).astype(np.uint8)
    return overall_mean


def create_dense_dataset(image: np.ndarray,
                         mask: np.ndarray,
                         max_copies: int = 7,
                         max_offset: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    create_dense_dataset 根据给定的稀疏目标图像创建稠密目标图像

    Args:
        image (np.ndarray): 图像数据
        mask (np.ndarray): 掩码数据
        max_copies (int, optional): 最大的稠密目标数量. Defaults to 7.
        max_offset (int, optional): 最大偏差. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回稠密目标的图像和掩码
    """
    dense_image = image.copy()
    dense_mask = mask.copy()
    min_num_copies = 3
    num_copies = random.randint(min_num_copies, max_copies)

    for _ in range(num_copies):
        # Find contours in the mask
        contours, _ = cv2.findContours(
            dense_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate random offsets for the copy
            min_offset = 7
            offset_x = random.randint(min_offset, max_offset)
            if random.randint(-1, 0) == -1:
                offset_x *= -1
            offset_y = random.randint(min_offset, max_offset)
            if random.randint(-1, 0) == -1:
                offset_y *= -1

            # Calculate the new top-left corner for the copy
            new_x = max(0, x + offset_x)
            new_y = max(0, y + offset_y)

            # Calculate the new bottom-right corner for the copy
            new_x2 = new_x + w
            new_y2 = new_y + h

            # Check if the new bottom-right corner is within image boundaries
            if new_x2 <= dense_image.shape[1] and new_y2 <= dense_image.shape[0]:
                # Calculate the dimensions of the overlapping region
                overlap_w = new_x2 - new_x
                overlap_h = new_y2 - new_y

                # Crop and copy the overlapping region
                roi_image = image[y:y+overlap_h, x:x+overlap_w]
                roi_mask = mask[y:y+overlap_h, x:x+overlap_w]
                dense_image[new_y:new_y + overlap_h,
                            new_x:new_x + overlap_w] = roi_image
                dense_mask[new_y:new_y + overlap_h,
                           new_x:new_x + overlap_w] = roi_mask

    return dense_image, dense_mask


# Folder paths
image_folder = r'data\image'  # Provide the path to the image folder
mask_folder = r'data\mask'    # Provide the path to the mask folder
output_folder = r'data\results\images'  # Provide the path to the output folder
mask_output_folder = r'data\results\masks'
# Process images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load image and mask
        image_path = os.path.join(image_folder, filename)
        # Assuming mask filenames match image filenames
        mask_path = os.path.join(mask_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Process large objects in the image and update the mask
        processed_image, processed_mask = shrink_large_objects(
            image.copy(), mask.copy())

        # Generate a dense dataset using the new function
        dense_image, dense_mask = create_dense_dataset(
            processed_image.copy(), processed_mask.copy(), max_copies=7, max_offset=10)

        # Save the dense dataset
        output_image_path = os.path.join(output_folder, filename)
        output_mask_path = os.path.join(
            mask_output_folder, filename)
        cv2.imwrite(output_image_path, dense_image)
        cv2.imwrite(output_mask_path, dense_mask)

# Display completion message
print('Processing complete.')
