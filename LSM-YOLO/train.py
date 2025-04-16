import warnings
warnings.filterwarnings('ignore')
import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
import re

torch.serialization.add_safe_globals([DetectionModel])

# --- 数据预处理函数 ---
def voc_to_yolo(xml_path, output_dir, img_width, img_height, class_names):
    # 检查文件是否存在
    if not os.path.exists(xml_path):
        print(f"File not found: {xml_path}. Skipping...")
        return  # 如果文件不存在，直接返回

    # 打印调试信息
    print(f"Processing XML file: {xml_path}")

    # 解析 XML 文件
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}. Skipping...")
        return

    # 获取 <filename> 标签的内容
    filename = root.find("filename").text

    # 使用正则表达式提取非数字部分
    match = re.match(r"([^\d]+)", filename)
    if match:
        filename_prefix = match.group(1).rstrip('_')  # 去掉可能的下划线
    else:
        filename_prefix = filename  # 如果没有匹配到，直接使用整个文件名

    # 打开输出文件
    output_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_path))[0] + ".txt")
    with open(output_file_path, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_id = class_names.index(filename_prefix)  # 从names列表获取类别ID
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            # 转换为YOLO格式（归一化中心坐标和宽高）
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 打印提取的文件名前缀
    print(f"Extracted filename prefix: {filename_prefix}")

def prepare_dataset():
    """划分数据集并转换标注格式"""
    # 创建目录
    os.makedirs("oxford-iiit-pet/images/train", exist_ok=True)
    os.makedirs("oxford-iiit-pet/images/val", exist_ok=True)
    os.makedirs("oxford-iiit-pet/labels/train", exist_ok=True)
    os.makedirs("oxford-iiit-pet/labels/val", exist_ok=True)

    # 读取trainval.txt划分训练集/验证集
    with open("oxford-iiit-pet/trainval.txt") as f:
        files = [line.strip().split()[0] for line in f.readlines()]

    # 划分数据集
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    # 处理训练集
    for file in train_files:
        img_path = f"oxford-iiit-pet/images/{file}.jpg"
        xml_path = f"oxford-iiit-pet/annotations/xmls/{file}.xml"
        # 复制图片
        shutil.copy(img_path, "oxford-iiit-pet/images/train/")
        # 转换XML标注为YOLO格式
        voc_to_yolo(
            xml_path=xml_path,
            output_dir="oxford-iiit-pet/labels/train",
            img_width=600,  # 假设图像尺寸为600x400
            img_height=400,
            class_names=names
        )

    # 处理验证集
    for file in val_files:
        img_path = f"oxford-iiit-pet/images/{file}.jpg"
        xml_path = f"xoxford-iiit-pet/annotations/xmls/{file}.xml"
        shutil.copy(img_path, "oxford-iiit-pet/images/val/")
        voc_to_yolo(
            xml_path=xml_path,
            output_dir="oxford-iiit-pet/labels/val",
            img_width=600,
            img_height=400,
            class_names=names
        )

# --- 类别列表（与data.yaml一致） ---
names = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
    'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
    'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
    'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]

if __name__ == '__main__':
    prepare_dataset()
    model = YOLO('ultralytics/cfg/models/LSM-YOLO/LSM-YOLO.yaml')
    model.train(data=r'C:\yolo\LSM-YOLO\oxford-iiit-pet\data.yaml',
                cache=False,
                project='runs/train',
                name='exp5',
                epochs=1,
                batch=48,
                close_mosaic=0,
                optimizer='SGD', # using SGD
                device='',
                # resume='', # last.pt path
                )