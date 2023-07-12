class DecoratorClass:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("Before function execution")
        result = self.func(*args, **kwargs)
        print("After function execution")
        return result


@DecoratorClass
def hello(name):
    print(f"Hello, {name}!")


# 调用被装饰的函数
hello("Alice")
