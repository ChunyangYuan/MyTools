"""Convert gt_seg_maps to gt_bboxes"""
import os
import mmcv
import skimage.measure as skm
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from lxml import etree

enlarge_flag = True

curr_dir = Path(__file__).parent.absolute()
data_root = 'data/sirst'
idx_file = os.path.expanduser(os.path.join(data_root, 'splits',
                                           'trainvaltest.txt'))
img_dir = os.path.expanduser(os.path.join(data_root, 'imgs'))
mask_dir = os.path.expanduser(os.path.join(data_root, 'masks'))
if enlarge_flag:
    bbox_dir = os.path.expanduser(os.path.join(data_root, 'enlarge_bboxes'))
else:
    bbox_dir = os.path.expanduser(os.path.join(data_root, 'bboxes'))
if not os.path.exists(bbox_dir):
    os.makedirs(bbox_dir)

# def save_plot_image(img, bboxes, idx):
#     plt.imshow(img, cmap='gray', vmin=0, vmax=255)
#     for xmin, ymin, xmax, ymax in bboxes:
#         # plt.plot(xc, yc, '+')
#         # TODO: plot bbox
#     plt.axis('off')
#     save_fig_path = os.path.expanduser(os.path.join(bbox_dir, idx + '.png'))
#     plt.savefig(save_fig_path, dpi=100, bbox_inches='tight', pad_inches=0)
#     plt.close()


def write_bbox_to_file(img, bboxes, idx):

    annotation = ET.Element('annotation')
    filename = ET.SubElement(annotation, 'filename')
    filename.text = idx

    _height, _width = img.shape
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(_width)
    height = ET.SubElement(size, 'height')
    height.text = str(_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(1)

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        label_object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(label_object, 'name')
        name.text = 'Target'

        _bbox = ET.SubElement(label_object, 'bbox')
        xmin_elem = ET.SubElement(_bbox, 'xmin')
        xmin_elem.text = str(xmin)

        ymin_elem = ET.SubElement(_bbox, 'ymin')
        ymin_elem.text = str(ymin)

        xmax_elem = ET.SubElement(_bbox, 'xmax')
        xmax_elem.text = str(xmax)

        ymax_elem = ET.SubElement(_bbox, 'ymax')
        ymax_elem.text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree_str = ET.tostring(tree.getroot(), encoding='unicode')
    save_xml_path = os.path.expanduser(
        os.path.join(bbox_dir, idx + '_bbox.xml'))
    root = etree.fromstring(tree_str).getroottree()
    root.write(save_xml_path, pretty_print=True)


def main():
    # load image and mask paths
    with open(idx_file, "r") as lines:
        # print("lines:", lines)
        for line in lines:
            idx = line.rstrip('\n')
            print(idx)
            _image = os.path.join(img_dir, idx + ".jpg")
            _mask = os.path.join(mask_dir, idx + ".png")

            img = mmcv.imread(_image, flag='grayscale')
            hei, wid = img.shape[:2]
            mask = mmcv.imread(_mask, flag='grayscale')
            label_img = skm.label(mask, background=0)
            regions = skm.regionprops(label_img)
            bboxes = []
            for region in regions:
                # convert starting from 0 to starting from 1
                ymin, xmin, ymax, xmax = np.array(region.bbox) + 1
                if enlarge_flag:
                    ymin -= 0
                    xmin -= 0
                    xmax += 0
                    ymax += 0

                # ymin, xmin are inside the region, but xmax and ymax not
                ymin -= 1
                xmin -= 1
                # boundary check
                xmin = max(1, xmin)
                ymin = max(1, ymin)
                xmax = min(wid, xmax)
                ymax = min(hei, ymax)
                bboxes.append([xmin, ymin, xmax, ymax])
            print(bboxes)
            # visualize
            # save_plot_image(img, bboxes, idx)

            # create xml for bboxes
            write_bbox_to_file(img, bboxes, idx)
            # break


if __name__ == '__main__':
    main()
