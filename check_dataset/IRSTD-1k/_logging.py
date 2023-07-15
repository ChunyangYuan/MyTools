import logging

# 创建名为 "ycy" 的日志记录器
logger = logging.getLogger("ycy")

# 配置日志记录器的日志级别
logger.setLevel(logging.INFO)

# 创建一个处理程序，将日志消息发送到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个文件处理程序，将日志消息写入文件
file_handler = logging.FileHandler("ycy.log")
file_handler.setLevel(logging.INFO)

# 创建一个格式化器，指定日志消息的格式
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# 为处理程序设置格式化器
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理程序添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 记录输出信息
logger.info("This is an informational message")
logger.warning("This is a warning message")
logger.error("This is an error message")
