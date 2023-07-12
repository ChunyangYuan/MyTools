# numbers = [[1, 2, 3, 4, 5]]
numbers = [1, 2, 3, 4, 5]
print(*numbers)  # 解包列表，打印元素
# 输出：1 2 3 4 5
print(*tuple(numbers))
first, *rest = numbers  # 解包列表，将第一个元素赋值给first，其余的元素赋值给rest列表
print(first)  # 输出：1
print(rest)   # 输出：[2, 3, 4, 5]

list1 = [1, 2, 3]
list2 = [4, 5, 6]
merged_list = [*list1, *list2]
print(merged_list)  # 输出：[1, 2, 3, 4, 5, 6]
