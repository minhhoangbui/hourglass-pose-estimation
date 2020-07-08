import xml.etree.ElementTree as ET
import glob
import os

# NOTE: remove xml file with no annotation
# src = '/home/hoang/datasets/JR/test/annos/'
# files = glob.glob(src + '*.xml')
#
# for file in files:
#     print(file)
#     tree = ET.parse(file)
#     root = tree.getroot()
#     if root.find('object') is None:
#         os.remove(file)

# NOTE: remove image with no annotation
root = '/home/hoang/datasets/JR/train'
images_dir = root + '/images/'
annos_dir = root + '/annos/'
xml_files = glob.glob(annos_dir + '*.xml')
img_files = glob.glob(images_dir + '*.jpg')
fname = []
for xml in xml_files:
    name = os.path.basename(xml)
    _name, _ = os.path.splitext(name)
    fname.append(_name)

for image in img_files:
    name = os.path.basename(image)
    _name, _ = os.path.splitext(name)
    if _name not in fname:
        os.remove(image)

