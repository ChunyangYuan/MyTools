import numpy as np

x = np.arange(25).reshape((5, 5))
print(x)

xx = x[:2, ...]
print(xx)
print(x[:2, :])
