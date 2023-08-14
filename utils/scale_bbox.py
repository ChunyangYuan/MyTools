import xml.etree.cElementTree as ET
import os
import os.path as osp
import copy


def BBox_2X(xml_folder: str, output_folder: str, scale: int = 2) -> None:
    """
    BBox_2X 批处理xml文件

    Args:
        xml_folder (str): xml文件存放文件夹路径
        output_folder (str): 保存文件夹路径
        scale (int, optional): 缩放因子. Defaults to 2.
    """
    if not osp.exists(xml_folder):
        print("xml_folder:{} do not exits!".format(xml_folder))
        sys.exit(1)

    if not osp.exists(output_folder):
        os.mkdir(output_folder)

    xml_list = os.listdir(xml_folder)
    for i in range(len(xml_list)):
        xml_path = osp.join(xml_folder, xml_list[i])
        xml_save_path = osp.join(output_folder, xml_list[i])
        scale_BBox_and_save(xml_path, xml_save_path, scale)
        print('{} file is scaled!'.format(xml_list[i]))


def scale_BBox_and_save(xml_path: str, save_path: str, scale: int = 2) -> None:
    """
    modify_BBox 修改xml文件中的BBox, 使其中图片宽高、bbox边框坐标值缩放scale倍

    Args:
        xml_path (str): 要修改的xml文件路径
        save_path (str): 修改后的xml文件保存路径
        scale (int, optional): 缩放因子. Defaults to 2.
    """
    # 解析 XML 文件
    tree = ET.parse(xml_path)

    # 获取根元素
    root = tree.getroot()

    # 遍历根元素的子元素
    for element in root:
        # 检查元素的标签名是否在目标元素列表中
        tag = element.tag
        if tag == 'size':
            for subelement in element:
                if subelement.tag == 'width' or subelement.tag == 'height':
                    # print(type(subelement.text))
                    subelement.text = str(2 * int(subelement.text))
        elif tag == 'object':
            for subelement in element:
                if subelement.tag == 'bndbox':
                    for subsubelement in subelement:
                        subsubelement.text = str(2 * int(subsubelement.text))
    # save modified xml
    tree.write(save_path)


if __name__ == "__main__":
    BBox_2X(r'E:\dataset\IRSTD-1k\Annotations',
            r'E:\dataset\IRSTD-1k\Annotations_2x')
