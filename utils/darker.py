import numpy as np
from PIL import Image


path = r'C:\Users\LwhYcy\Desktop\U\slic\200.jpg'
image = Image.open(path)
image_np = np.array(image)
image_np = np.where(image_np >= 233, 0, image_np)
min = np.min(image_np)
max = np.max(image_np)
# image_np -= min
image = Image.fromarray(image_np)
image.save('darker.png')
pass
