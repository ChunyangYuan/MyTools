class DecoratorClass:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print(f"Decorator arguments: {self.arg1}, {self.arg2}")
            print("Before function execution")
            result = func(*args, **kwargs)
            print("After function execution")
            return result
        return wrapper


@DecoratorClass("Hello", 42)
def example_function(arg1, arg2, **kwargs):
    print(f"arg1: {arg1}")
    print(f"arg2: {arg2}")
    for key, value in kwargs.items():
        print(f"{key}: {value}")


# 调用被装饰的函数
example_function("Hello", 42, name="Alice", city="New York")
