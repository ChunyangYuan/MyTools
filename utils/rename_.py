import os

# 原始文件夹路径
folder_path = "path/to/your/folder"

# 获取文件夹中所有文件的列表
file_list = os.listdir(folder_path)

# 遍历文件列表
for filename in file_list:
    # 判断是否为图片文件（可以根据您的需求添加更多的图片格式，如.jpg、.jpeg等）
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        # 修改文件名（假设要将"_0"替换为空字符串）
        new_filename = filename.replace("_0", "")

        # 原文件的完整路径
        old_file_path = os.path.join(folder_path, filename)

        # 新文件的完整路径
        new_file_path = os.path.join(folder_path, new_filename)

        # 将文件重命名并保存至原来的文件夹中
        os.rename(old_file_path, new_file_path)

        print(f"文件名已修改为: {new_filename}")

print("所有图片文件名修改完成。")
