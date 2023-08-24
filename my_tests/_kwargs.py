# def outer_function(**kwargs):
#     def inner_function(img='', mask='', ann=''):
#         # 在这里执行你的内部函数逻辑
#         # 使用 img、mask 和 ann 参数进行操作
#         print('ok')
#         pass

#     if 'img' in kwargs and 'mask' in kwargs and 'ann' in kwargs:
#         inner_function(kwargs)
#     else:
#         print("Missing required parameters: img, mask, ann")


# # 使用示例
# parameters = {
#     'img': 'image_data',
#     'mask': 'mask_data',
#     'ann': 'annotation_data',
#     'label': 'label'
# }

# outer_function(**parameters)


class ParentClass:
    def __init__(self, img='', mask='', ann=''):
        self.img = img
        self.mask = mask
        self.ann = ann

    def process(self):
        # 在这里执行父类的处理逻辑，使用 self.img、self.mask 和 self.ann
        pass


class ChildClass(ParentClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.child_process()

    def child_process(self):
        print('ok')
        # 在这里执行子类的处理逻辑，可以使用继承的属性 self.img、self.mask 和 self.ann
        pass


# 使用示例
child_params = {
    'img': 'image_data',
    'mask': 'mask_data',
    'ann': 'annotation_data',
    'label': 'label'
}

child_instance = ChildClass(**child_params)
# child_instance.process()  # 调用父类方法
# child_instance.child_process()  # 调用子类方法
