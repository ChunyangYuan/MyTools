import xml.etree.cElementTree as ET

# 解析 XML 文档
tree = ET.parse(r'my_tests\xml_demo.xml')

# 获取根元素
root = tree.getroot()

# 遍历 book 元素
for book in root:
    # 获取元素的标签名
    print("Element:", book.tag)

    # 获取元素的属性
    attributes = book.attrib
    print("Attributes:", attributes)

    # # 获取特定属性的值
    # isbn = book.attrib["isbn"]
    # print("ISBN:", isbn)
