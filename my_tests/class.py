class AllTypesClass:
    static_variable = "Static Variable"
    _protected_variable = "Protected Variable"
    __private_variable = "Private Variable"

    def __init__(self):
        self.instance_variable = "Instance Variable"
        self._protected_instance_variable = "Protected Instance Variable"
        self.__private_instance_variable = "Private Instance Variable"

    @staticmethod
    def static_function():
        print("This is a static function.")

    def instance_function(self):
        print("This is an instance function.")
        self.__private_function()

    # @staticmethod
    # def _protected_function1(self):
    #     print("This is a protected function.")

    def _protected_function2(self):
        print("This is a protected function.")

    # @staticmethod
    # def __private_function1(self):
    #     print("This is a private function.")

    def __private_function2(self):
        print("This is a private function.")


# 访问静态变量
print(AllTypesClass.static_variable)

# 访问保护变量
print(AllTypesClass._protected_variable)

# 访问私有变量
# 注意：私有变量名会被 Python 解释器修改，添加一个前缀 "_类名"，即 "_AllTypesClass__private_variable"
print(AllTypesClass._AllTypesClass__private_variable)

# 调用静态函数
AllTypesClass.static_function()

# 创建对象
obj = AllTypesClass()

# 访问实例变量
print(obj.instance_variable)

# 访问保护实例变量
print(obj._protected_instance_variable)

# 访问私有实例变量
# 同样，私有变量名会被 Python 解释器修改，添加一个前缀 "_类名"
print(obj._AllTypesClass__private_instance_variable)

# 调用实例函数
obj.instance_function()

# 调用保护函数
obj._protected_function()
