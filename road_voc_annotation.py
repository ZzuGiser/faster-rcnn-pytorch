import xml.etree.ElementTree as ET
from os import getcwd

sets=[('road', 'train')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes =["road"]
def convert_annotation(image_id, list_file):
    in_file = open('road_val_data/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
            
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


for class_name, image_set in sets:
    image_ids = open('road_val_data/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(class_name, image_set), 'w')
    for image_id in image_ids:
        list_file.write('./road_val_data/photo/%s.jpg'%(image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
