import xml.etree.ElementTree as ET
tree = ET.parse('p04.rsml')
root = tree.getroot()
for child in root:
    for childchild in child:
        print(childchild.tag, childchild.attrib, childchild.text)
import pdb;  pdb.set_trace()