from PIL import Image
import os


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size[0], img.size[1]


def find_max_min_dimensions(folder_path):
    max_width, max_height = float('-inf'), float('-inf')
    min_width, min_height = float('inf'), float('inf')

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            width, height = get_image_dimensions(image_path)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            min_width = min(min_width, width)
            min_height = min(min_height, height)

    return max_width, max_height, min_width, min_height


if __name__ == "__main__":
    folder_path = "E:\dataset\SIRSTdevkit-master\PNGImages"
    max_width, max_height, min_width, min_height = find_max_min_dimensions(
        folder_path)
    print(f"Max Width: {max_width}, Max Height: {max_height}")
    print(f"Min Width: {min_width}, Min Height: {min_height}")
