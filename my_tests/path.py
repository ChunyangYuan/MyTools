import os
from pathlib import Path
# 获取当前脚本的文件路径
# print(__file__)
# print(Path(__file__))
# print(Path(__file__).parent.absolute())
# script_path = os.path.abspath(__file__)
data_root = 'data/sirst'
idx_file = os.path.expanduser(os.path.join(data_root, 'splits',
                                           'trainvaltest.txt'))
# 打印文件路径
print("Script Path:", idx_file)
