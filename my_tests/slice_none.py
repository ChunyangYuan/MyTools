import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# 在第一个位置插入轴
result1 = arr[None, :]
print(result1.shape)  # 输出 (1, 2, 3)

# 在第二个位置插入轴
result2 = arr[:, None]
print(result2.shape)  # 输出 (2, 1, 3)

# 在第三个位置插入轴
result3 = arr[:, :, np.newaxis]
print(result3.shape)  # 输出 (2, 3, 1)

# 在第三个位置插入轴
result4 = arr[:, :, None]
print(result4.shape)  # 输出 (2, 3, 1)

# 在第三个位置插入轴
result5 = arr[..., None]
print(result5.shape)  # 输出 (2, 3, 1)
