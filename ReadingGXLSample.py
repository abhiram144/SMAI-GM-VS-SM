import xml.etree.ElementTree as ET

tree = ET.parse("filename")

for edge in tree.findall(".//edge"):
    #node_value = node.text
    start = edge.get('from')
    end = edge.get('to')