
def calculate_mean_around_box(image: np.ndarray,
                              x: int,
                              y: int,
                              width: int,
                              height: int) -> numpy.uint8:
    """
    calculate_mean_around_box 以给定坐标框为中心,计算其八邻域像素均值

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
