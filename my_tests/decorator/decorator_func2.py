def decorator_function(func):
    def wrapper(*args, **kwargs):
        print("Before function execution")
        result = func(*args, **kwargs)
        print("After function execution")
        return result
    return wrapper


@decorator_function
def example_function(arg1, arg2, **kwargs):
    print(f"arg1: {arg1}")
    print(f"arg2: {arg2}")
    for key, value in kwargs.items():
        print(f"{key}: {value}")


# 调用被装饰的函数
example_function("Hello", 42, name="Alice", city="New York")
