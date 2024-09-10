import xml.etree.ElementTree as ET

file_path = 'assets/desk_multi2.xml'
tree = ET.ElementTree(file=file_path)

num = 8
width = 0.005

for i in range(-num+1,num):

    for j in range(-num+1,num):
        pos = '0.005 {} {}'.format((i*width),(j*width))
        tactile = ET.Element('body')
        tactile.attrib = {'pos': pos, 'euler': '0.0 0.0 0'}
        geom1 = ET.SubElement(tactile,'geom')
        geom1.attrib = {'class': 'robot0:D_Contact', 'friction': '0.5 0.5 0.005', 'mesh': 'tactile_sphere2', 'pos': '-0.0005 0 0', 'quat': '0.707107 0 0.707107 0'}
        geom2 = ET.SubElement(tactile,'geom')
        geom2.attrib = {'class': 'robot0:Tactile', 'friction': '0.5 0.5 0.005', 'mesh': 'tactile_sphere2', 'pos': '-0.0005 0 0', 'quat': '0.707107 0 0.707107 0'}

        node = tree.find('body/body/body')
        node.append(tactile)
        tree.write(file_path)