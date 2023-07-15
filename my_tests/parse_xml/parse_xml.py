try:
    import xml.etree.cElementTree as ET
except ImportError:
    print("xml.etree.ElementTree")
    import xml.etree.ElementTree as ET
from lxml import etree
import os
print(os.getcwd())

tree = ET.parse(r'my_tests\Misc_1.xml')
root = tree.getroot()


for element in root:
    print(element.tag, element.attrib, element.text)
    for child in element:
        print(child.tag, child.text)
