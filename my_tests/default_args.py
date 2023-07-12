# TODO 默认参数
def defaultzero(list=[]):  # 我们的本意是提供的list参数为0时 返回只有一个0的list
    list.append(0)
    return list


print(defaultzero())

print(defaultzero())
# 结果说明python解释器会将默认参数作为一个公共对象来对待，多次调用含有默认参数的函数，就会进行多次修改。
#  因此定义默认参数时一定要使用不可变对象(int、float、str、tuple)。使用可变对象语法上没错，但在逻辑上是不安全的，代码量非常大时，容易产生很难查找的bug。

print("="*58)

# TODO 可变参数


def getsum(*num):
    print(type(num))
    print(num)
    sum = 0
    for n in num:
        sum += n
    return sum


list = [2, 3, 4]

print(getsum(1, 2, 3))
print(getsum(*list))
# TypeError: unsupported operand type(s) for +=: 'int' and 'tuple'
# print(getsum((1, 2, 3,)))
# 结果：6 9
print("="*58)

# TODO 关键字参数


def personinfo(name, age, **kw):
    print('name:', name, 'age:', age, 'ps:', kw)


personinfo('Steve', 22)
personinfo('Lily', 23, city='Shanghai')
personinfo('Leo', 23, gender='male', city='Shanghai')


print("="*58)
# TODO 命名关键字参数


def personinfo2(name, age, *, gender, city):  # 只能传递gender和city参数
    print(name, age, gender, city)


personinfo2('Steve', 22, gender='male', city='shanghai')

# TODO 各种参数之间组合
# 一次函数调用可以传递以上所述任何一种参数或者多种参数的组合，当然也可以没有任何参数。正如默认参数必须在最右端一样，使用多种参数时也对顺序有严格要求，也是为了解释器可以正确识别到每一个参数。

# TODO 顺序：基本参数、默认参数、可变参数、命名关键字参数和关键字参数。
