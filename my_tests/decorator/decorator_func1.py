def decorator_function(func):
    def wrapper():
        print("Before function execution")
        func()
        print("After function execution")
    return wrapper


@decorator_function
def hello():
    print("Hello, world!")


# 调用被装饰的函数
hello()
