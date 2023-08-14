# 定义要写入文件的内容
content = "Hello, World!"

# 定义文件名
file_name = "example.txt"

# 使用with语句打开文件，确保在处理完文件后自动关闭
with open(file_name, "w") as file:
    # 写入内容到文件
    file.write(content)

# 提示写入成功
print("写入文件成功！")
