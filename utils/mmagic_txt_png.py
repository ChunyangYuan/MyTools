def add_png_extension(input_txt, output_txt):
    with open(input_txt, 'r') as input_file, open(output_txt, 'w') as output_file:
        for line in input_file:
            line = line.strip()  # 去除行末尾的换行符和空白字符
            line_with_extension = f"{line}.png\n"
            output_file.write(line_with_extension)


if __name__ == "__main__":
    input_txt_file = r"E:\dataset\SIRSTdevkit-master\Splits\trainvaltest.txt"    # 替换成输入的txt文件路径
    output_txt_file = r"E:\dataset\SIRSTdevkit-master\trainvaltest.txt"  # 替换成输出的txt文件路径

    add_png_extension(input_txt_file, output_txt_file)
