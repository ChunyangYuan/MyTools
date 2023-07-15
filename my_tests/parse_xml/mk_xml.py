import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom

# 创建根元素
root = ET.Element("root")

# 创建子元素
child1 = ET.SubElement(root, "child1")
child1.text = "Hello"

child2 = ET.SubElement(root, "child2")
child2.text = "World"

# 创建 XML 树
tree = ET.ElementTree(root)

# 保存为 XML 文件
tree.write("output.xml")

# 使用 minidom 进行美化
dom = minidom.parse("output.xml")
with open("formatted_output.xml", "w") as file:
    file.write(dom.toprettyxml())
