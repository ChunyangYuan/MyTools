class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @width.setter
    def width(self, value):
        if value > 0:
            self._width = value
        else:
            raise ValueError("Width must be positive.")

    @height.setter
    def height(self, value):
        if value > 0:
            self._height = value
        else:
            raise ValueError("Height must be positive.")

    def area(self):
        return self._width * self._height


# 创建对象
rect = Rectangle(5, 3)

# 访问属性
print(rect.width)  # 输出: 5
print(rect.height)  # 输出: 3

# 调用方法
print(rect.area())  # 输出: 15

# 设置属性
rect.width = 8
rect.height = 4

# 再次访问属性
print(rect.width)  # 输出: 8
print(rect.height)  # 输出: 4
print(rect.area())  # 输出: 32

# 尝试设置非正数值
rect.width = -2  # 抛出异常: ValueError: Width must be positive.
