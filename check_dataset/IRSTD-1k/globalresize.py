
def shrink_large_objects(image: np.ndarray,
                         mask: np.ndarray,
                         mode: str = 'globalresize',
                         object_max_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    shrink_large_objects 缩小大目标

    Args:
        image (np.ndarray): 图像数据
        mask (np.ndarray): 掩码数据
        mode (str) : 'globalresize','localresize' 两种模式,一种是'globalresize'模式,即resize整个图像,
        再添加padding为原始图像的尺寸(特点是缩小的目标处更加平滑自然);
        第二种是'localresize',采用局部resize,只对目标区域进行resize,再经过滤波腐蚀操作来缩小目标尺寸
        (特点是无需添加padding,只不过缩小的目标处放大后有些许人工处理痕迹)
        object_max_size (int, optional): 目标最大尺寸. Defaults to 7.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回处理后的图像和掩码
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                # mask[max(
                #     0, original_object_y1 - padding):min(img_height, original_object_y1 + original_object_h + padding), max(0, original_object_x1 - padding):min(img_width, original_object_x1 + original_object_w + padding)] = blurred_mask
                # 增强图像目标
                # 定义亮度增加的值
                # brightness_increase = 50

                # # 使用布尔索引选择满足条件的像素位置
                # selected_pixels = mask == 255

                # # 在满足条件的位置上增加亮度,同时确保不超过255
                # # image[selected_pixels] = np.minimum(
                # #     image[selected_pixels] + brightness_increase, 230)
                # adjusted_pixels = image[selected_pixels] + brightness_increase
                # adjusted_pixels[adjusted_pixels > 255] = 255
                # image[selected_pixels] = adjusted_pixels
    elif mode == 'globalresize':
        max_w = max_h = 0
        for contour in contours:
            # Calculate the bounding rectangle for the contour
            original_object_x1, original_object_y1, original_object_w, original_object_h = cv2.boundingRect(
                contour)

            if original_object_w >= object_max_size or original_object_h >= object_max_size:
                max_w = max(max_w, original_object_w)
                max_h = max(max_h, original_object_h)
        if max_w == 0 or max_h == 0:
            return image, mask

        # resize
        # Calculate the scaling factor to shrink to target size
        scale_factor = min(object_max_size / max_w, object_max_size / max_h)
        print('scale_factor:{}'.format(scale_factor))
        # Calculate the new width and height
        new_w = int(image.shape[1] * scale_factor)
        new_h = int(image.shape[0] * scale_factor)
        # Ensure that both the new size and the original size are either odd or even.
        if image.shape[1] % 2 == 0:
            new_w = new_w if new_w % 2 == 0 else new_w - 1
        else:
            new_w = new_w if new_w % 2 != 0 else new_w - 1
        if image.shape[0] % 2 == 0:
            new_h = new_h if new_h % 2 == 0 else new_h - 1
        else:
            new_h = new_h if new_h % 2 != 0 else new_h - 1

        print('new_w:{}'.format(new_w))
        print('new_h:{}'.format(new_h))
        # Resize the entire image
        resized_image = cv2.resize(image, (new_w, new_h))
        resized_mask = cv2.resize(mask, (new_w, new_h))
        # Calculate padding
        pad_x = (image.shape[1] - new_w) // 2
        pad_y = (image.shape[0] - new_h) // 2

        # Create a padded image
        image = cv2.copyMakeBorder(
            resized_image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
        mask = cv2.copyMakeBorder(
            resized_mask, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)

    else:
        raise ValueError(
            "mode support 'localresize' and 'globalresize'")

    return image, mask
