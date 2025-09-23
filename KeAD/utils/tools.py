import os
import sys
import time
import json
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from collections import defaultdict
import subprocess
import re
import torch
import numpy as np
import random
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 随机选择文件
def select_random_file(dir_path):
    files = os.listdir(dir_path)
    return random.choice(files)


import random
 
 
def data_split(full_list, ratio=None, offset=None, shuffle=True):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    assert ratio is not None or offset is not None
    n_total = len(full_list)
    if offset is None:
        offset = int(n_total * ratio)
    else:
        offset = min(offset, n_total)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def display_images_and_masks(image_paths, mask_paths, titles, ncols=2):
    """
    显示图像和对应的分割mask。
    :param image_paths: 原图的路径列表。
    :param mask_paths: 分割mask的路径列表。
    :param titles: 每种类型的标题列表。
    :param ncols: 每行显示的列数（默认为2）。
    """
    assert len(image_paths) == len(mask_paths) == len(titles), "图像、mask和标题的数量必须相等。"
    
    nrows = len(image_paths)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5*nrows))
    
    for i in range(nrows):
        # 读取并显示原图
        img = mpimg.imread(image_paths[i])
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'{titles[i]} - Image')
        axes[i, 0].axis('off')
        
        # 读取并显示分割mask
        mask = mpimg.imread(mask_paths[i])
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'{titles[i]} - Label')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


'''
def masks_to_boxes(masks: Tensor, anomaly_maps: Tensor | None = None) -> tuple[list[Tensor], list[Tensor]]:
    """Convert a batch of segmentation masks to bounding box coordinates.

    Args:
        masks (Tensor): Input tensor of shape (B, 1, H, W), (B, H, W) or (H, W)
        anomaly_maps (Tensor | None, optional): Anomaly maps of shape (B, 1, H, W), (B, H, W) or (H, W) which are
            used to determine an anomaly score for the converted bounding boxes.

    Returns:
        list[Tensor]: A list of length B where each element is a tensor of shape (N, 4) containing the bounding box
            coordinates of the objects in the masks in xyxy format.
        list[Tensor]: A list of length B where each element is a tensor of length (N) containing an anomaly score for
            each of the converted boxes.
    """
    height, width = masks.shape[-2:]
    masks = masks.view((-1, 1, height, width)).float()  # reshape to (B, 1, H, W) and cast to float
    if anomaly_maps is not None:
        anomaly_maps = anomaly_maps.view((-1,) + masks.shape[-2:])

    if masks.is_cuda:
        batch_comps = connected_components_gpu(masks).squeeze(1)
    else:
        batch_comps = connected_components_cpu(masks).squeeze(1)

    batch_boxes = []
    batch_scores = []
    for im_idx, im_comps in enumerate(batch_comps):
        labels = torch.unique(im_comps)
        im_boxes = []
        im_scores = []
        for label in labels[labels != 0]:
            y_loc, x_loc = torch.where(im_comps == label)
            # add box
            box = Tensor([torch.min(x_loc), torch.min(y_loc), torch.max(x_loc), torch.max(y_loc)]).to(masks.device)
            im_boxes.append(box)
            if anomaly_maps is not None:
                im_scores.append(torch.max(anomaly_maps[im_idx, y_loc, x_loc]))
        batch_boxes.append(torch.stack(im_boxes) if im_boxes else torch.empty((0, 4), device=masks.device))
        batch_scores.append(torch.stack(im_scores) if im_scores else torch.empty(0, device=masks.device))

    return batch_boxes, batch_scores


def boxes_to_masks(boxes: list[Tensor], image_size: tuple[int, int]) -> Tensor:
    """Convert bounding boxes to segmentations masks.

    Args:
        boxes (list[Tensor]): A list of length B where each element is a tensor of shape (N, 4) containing the bounding
            box coordinates of the regions of interest in xyxy format.
        image_size (tuple[int, int]): Image size of the output masks in (H, W) format.

    Returns:
        Tensor: Tensor of shape (B, H, W) in which each slice is a binary mask showing the pixels contained by a
            bounding box.
    """
    masks = torch.zeros((len(boxes),) + image_size).to(boxes[0].device)
    for im_idx, im_boxes in enumerate(boxes):
        for box in im_boxes:
            x_1, y_1, x_2, y_2 = box.int()
            masks[im_idx, y_1 : y_2 + 1, x_1 : x_2 + 1] = 1
    return masks
'''


import os
import random

def generate_train_val_lists(images_dir, save_dir, train_ratio=0.8):
    """
    从images目录中读取图像文件名，并随机生成train.txt和val.txt文件列表。
    :param images_dir: 包含图像的目录路径。
    :param train_ratio: 分配给训练集的比例。
    """
    # 获取所有图像文件名
    images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    # 随机打乱图像文件名
    random.shuffle(images)
    
    # 计算训练集大小
    train_size = int(len(images) * train_ratio)
    
    # 分割训练集和验证集
    train_images = images[:train_size]
    val_images = images[train_size:]
    
    # 保存训练集文件名到train.txt
    with open(os.path.join(save_dir, 'train.txt'), 'w') as f:
        for item in train_images:
            f.write("%s\n" % item)
    
    # 保存验证集文件名到val.txt
    with open(os.path.join(save_dir, 'val.txt'), 'w') as f:
        for item in val_images:
            f.write("%s\n" % item)
    print('generate_train_val_lists done train: ', len(train_images), ' -- val: ', len(val_images))


from PIL import Image


from PIL import Image
import os, re
#from xpinyin import Pinyin

# 提取中文
def extract_chinese(sentence):
    pattern=re.compile("[\u4e00-\u9fa5]+")
    results = pattern.findall(sentence)   
    return results


# 基础清洗数据，两个功能
# （1） 名字中带有中文或者空格括号的，空格括号使用 _ 代替，中文使用对应英文代替
#  (2) 删除带有指定字符的, 校验一下是否去除这个字符串，还是有中文
def clean_image_base(images_dir, english2chinese={}, delete_patterns=[], open=False):
    all_ch = set()
    pinyin_converter = None # Pinyin()
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.txt')):
                # check delete_patterns
                if len(delete_patterns) > 0:
                    for dp in delete_patterns:
                        if dp in file:
                            if not os.path.exists(os.path.join(root, file)):
                                continue
                            non_dp = file.split(dp)[0]+'.'+file.split('.')[-1]
                            if os.path.exists(os.path.join(root, non_dp)):
                                print('delete this file! ', os.path.join(root, file), '  for delete pattern ', dp)
                                if open:
                                    os.remove(os.path.join(root, file))
                            else:
                                print('non_dp is not exists! ', non_dp, ' -- ', dp)
                # 空格括号处理
                new_file_name = file.replace('(', '_').replace('（', '_').replace(')', '').replace('）', '').replace(' ', '').replace('__', '_')
                # check chinese
                chinese_list = extract_chinese(new_file_name)
                for ch in chinese_list:
                    if ch in english2chinese:
                        ch_pinyin = english2chinese[ch]
                    else:
                        ch_pinyin = pinyin_converter.get_pinyin(ch)
                    new_file_name = new_file_name.replace(ch, ch_pinyin)
                    all_ch.add(ch)
                
                if new_file_name != file:
                    print('rename this file! ', os.path.join(root, file), ' to ', os.path.join(root, new_file_name))
                    if open:
                        os.rename(os.path.join(root, file), os.path.join(root, new_file_name))
            else:
                print('unexpected data! ', file)
    print('all_ch ', all_ch)


def calculate_image_sizes_stats(images_dir):
    sizes = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    if 'Ba' in os.path.join(root, file) or 'ba' in os.path.join(root, file):
                        continue
                    with Image.open(os.path.join(root, file)) as img:
                        sizes.append(img.size)  # img.size is a tuple (width, height)
                except Exception as e:
                    print(f"Error opening {file}: {e}")

    if not sizes:
        print("没有找到任何图片文件。")
        return

    # 计算最大值、最小值和平均值
    max_size = max(sizes, key=lambda x: x[0]*x[1])
    min_size = min(sizes, key=lambda x: x[0]*x[1])
    avg_size = tuple(int(sum(x) / len(sizes)) for x in zip(*sizes))

    print(f"最大: {max_size}")
    print(f"最小: {min_size}")
    print(f"平均: {avg_size}")


def get_image_size(image_path):
    """
    获取图像的尺寸。
    :param image_path: 图像文件的路径。
    :return: 图像的尺寸，格式为(width, height)。
    """
    with Image.open(image_path) as img:
        return img.size

import numpy as np
from PIL import Image

def yolo_to_mask(yolo_path, image_size):
    """
    将YOLO格式的边界框转换为分割掩码。
    :param yolo_path: YOLO格式边界框文件的路径。
    :param image_size: 原始图像的尺寸，格式为(width, height)。
    :return: 分割掩码图像。
    """
    width, height = image_size
    # 初始化掩码图像，所有像素初始为0
    mask = np.zeros((height, width), dtype=np.uint8)

    # 读取YOLO格式的边界框
    with open(yolo_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, w, h = map(float, line.split())
            class_id = int(class_id)

            # 将YOLO格式转换为像素坐标
            x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
            x_min, y_min = int(x_center - w / 2), int(y_center - h / 2)
            x_max, y_max = int(x_center + w / 2), int(y_center + h / 2)

            # 在掩码上绘制边界框
            mask[y_min:y_max, x_min:x_max] = 1

    return mask

def nms(boxes, scores, iou_threshold):
    """
    非极大值抑制（Non-Maximum Suppression, NMS）。
    
    :param boxes: 边界框数组，每个元素为[x_min, y_min, x_max, y_max]。
    :param scores: 每个边界框对应的置信度得分。
    :param iou_threshold: IoU阈值。
    :return: 保留下来的边界框的索引。
    """
    # 计算每个边界框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 按得分从高到低排序所有边界框的索引
    order = scores.argsort()[::-1]

    keep = []  # 用于保存最后保留下来的边界框的索引
    while order.size > 0:
        i = order[0]  # 当前得分最高的边界框索引
        keep.append(i)
        # 计算当前得分最高的边界框与其他所有边界框的IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # 保留IoU小于阈值的边界框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def compute_iou_min(box1, box2):
    """
    计算两个框的IoU。
    """
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    #iou = inter_area / float(box1_area + box2_area - inter_area)
    iou = inter_area / float(min(box1_area, box2_area))
    return iou

def merge_overlapping_boxes(boxes, iou_threshold):
    """
    合并重叠的框。
    """
    merged_boxes = []
    while boxes:
        base_box = boxes.pop(0)
        to_merge = [base_box]
        for _ in range(len(boxes)):
            box = boxes.pop(0)
            #print('base_box ', base_box, ' box ', box, ' --> iou ', compute_iou_min(base_box, box))
            #print('iou ', compute_iou_min(base_box, box))
            if compute_iou_min(base_box, box) > iou_threshold:
                to_merge.append(box)
            else:
                boxes.append(box)
        # 合并框
        if len(to_merge) > 1:
            x_min = min(box[0] for box in to_merge)
            y_min = min(box[1] for box in to_merge)
            x_max = max(box[2] for box in to_merge)
            y_max = max(box[3] for box in to_merge)
            boxes.append([x_min, y_min, x_max, y_max])
        else:
            merged_boxes.append(base_box)
    return merged_boxes


def mask_to_yolo(mask_image_path, class_id, yolo_txt):
    '''
    将分割掩码转为YOLO格式的边界框, 需要考虑一张图像中存在多个缺陷区域的情况
    mask_image: 掩码图像
    class_id:  类别id 针对实例分割情况需要单独考虑，一般是一个列表代表一种缺陷
    yolo_txt: 保存yolo标注文件
    '''
    image_width, image_height = get_image_size(mask_image_path)
    yolo_annotations = []
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    # 查找轮廓
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    with open(yolo_txt, 'w') as wf:
        for contour in contours:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
            boxes.append([x_min, y_min, x_max, y_max])
    
        merged_boxes =  boxes  #merge_overlapping_boxes(boxes, 0.0001)
        last_num = len(boxes)
        this_num = 0
        merged_boxes = boxes
        while this_num < last_num:
            last_num = len(merged_boxes)
            merged_boxes = merge_overlapping_boxes(merged_boxes, 0.1)
            this_num = len(merged_boxes)

        for box in merged_boxes:
            x_min, y_min, x_max, y_max = box
            x, y, w, h = x_min, y_min, x_max-x_min, y_max-y_min
            # 转换为YOLO格式
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            width = w / image_width
            height = h / image_height
            # 假设目标类别为0
            yolo_annotation = f"{class_id} {x_center} {y_center} {width} {height}"
            yolo_annotations.append(yolo_annotation)
            wf.write(yolo_annotation+'\n')
    return yolo_annotations
    

# 文件数量统计
def count_files_in_folders(dir):
    cate2cnt = {}
    cls2cnt = {}
    for root, dirs, files in os.walk(dir):
        root_split = root.split('/')
        cate = root_split[-1]
        cls = root_split[-2]
        if cate not in cate2cnt:
            cate2cnt[cate] = defaultdict(int)
        if cls not in cls2cnt:
            cls2cnt[cls] = defaultdict(int)

        file_count = defaultdict(int)
        for file in files:
            if '_resize.xml' in file or '_recover.xml' in file:
                continue
            ext = os.path.splitext(file)[-1]
            file_count[ext] += 1
            cate2cnt[cate][ext] += 1
            cls2cnt[cls][ext] += 1
        print(f"在目录 {root} 中:")
        for ext, count in file_count.items():
            print(f"后缀为{ext}的文件数量: {count}")
        print()
        
    for cate in cate2cnt:
        print('在目录后缀为: ', cate, ' 中')
        for ext, count in cate2cnt[cate].items():
            print(f"后缀为{ext}的文件数量: {count}")
        
    for cate in cls2cnt:
        print('在目录后缀为: ', cate, ' 中')
        for ext, count in cls2cnt[cate].items():
            print(f"后缀为{ext}的文件数量: {count}")
        

# 文件数量统计
def show_files_in_folders(dir, prefix=''):
    print(prefix+'-- ', dir+':')
    cls2cnt = {}
    prefix += '  '
    for file in os.listdir(dir):
        real_file = os.path.join(dir, file)
        if not os.path.isfile(real_file):
            
            cls2cnt_next = show_files_in_folders(real_file , prefix=prefix)
            for cls in cls2cnt_next:
                if cls not in cls2cnt:
                    cls2cnt[cls] = cls2cnt_next[cls]
                else:
                    cls2cnt[cls] += cls2cnt_next[cls]
            continue
        if '_resize.xml' in file or '_recover.xml' in file or file.startswith('._'):
            continue
        ext = os.path.splitext(file)[-1]
        if ext not in cls2cnt:
            cls2cnt[ext] = 0
        cls2cnt[ext] += 1

    for ext, count in cls2cnt.items():
        print(f"{prefix}-- 后缀为{ext}的文件数量: {count}")

    return cls2cnt


def resize_image(img_path, scale_percent):
    img = cv2.imread(img_path)  # 读取图像并转为灰度图
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100) 
    print('ori img ', img.shape)
    return cv2.resize(img, (width, height))


# 严格控制变量
def setup_seed(seed: int) -> None:
    '''
        Set random seed to make experiments repeatable
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True  # implement same config in cpu and gpu
    torch.backends.cudnn.benchmark = False


# 测量当前空闲gpu
def get_free_gpu():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free", "--format=csv,noheader,nounits"])
        gpu_infos = result.decode('utf-8').strip().split('\n')
        free_gpus = []
        for gpu_info in gpu_infos:
            index, gpu_util, mem_free = map(int, re.split(',\s*', gpu_info))
            if gpu_util < 10 and mem_free > 1000:  # You can change these thresholds
                free_gpus.append(index)
        return free_gpus
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

# 计时器
class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, update=False, info=''):
        if self.start_time is None:
            raise Exception("Timer is not started")
        elapsed_time = time.time() - self.start_time
        if update:
            self.start_time = time.time()
        print(f"Elapsed time: {elapsed_time} seconds. {info}")



def modify_class_name(xml_file_path, old_class_name, new_class_name):
    # 解析XML文件
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # 遍历所有的object标签
    for obj in root.findall('object'):
        # 查找name标签
        name = obj.find('name')
        if name.text == old_class_name:
            # 替换类名
            name.text = new_class_name
    
    # 保存修改后的XML文件
    tree.write(xml_file_path)


# 目标检测统计器
class Statistics:
    '''
    stats = Statistics()
    for xml_file in os.listdir('path_to_your_xml_files'):
        if xml_file.endswith('.xml'):
            stats.update(os.path.join('path_to_your_xml_files', xml_file))
    stats.report()
    '''
    def __init__(self):
        self.total_boxes = 0
        self.total_images = 0
        self.total_background_images = 0
        self.stats = defaultdict(lambda: {'count': 0, 'widths': [], 'heights': [], 'ratios': []})
        #label_list = ['spalling', 'rust', 'crack', 'discolor', 'blister']
        self.class2id = {}
        self.id2class = {}
    
    def set_labels(self, label_list):
        self.label_list = label_list
        self.class2id = {x:str(i) for i,x in enumerate(label_list)}
        self.id2class = {str(i):x for i,x in enumerate(label_list)}

    def update(self, info, is_xml=True):
        if info is None:
            self.total_background_images += 1
            return
        if is_xml:
            xml_path = info
            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 遍历每个object标签
            for obj in root.findall('object'):
                # 获取标注框的坐标
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # 计算标注框的宽度和高度
                width = xmax - xmin
                height = ymax - ymin

                # 获取类别名称
                class_name = obj.find('name').text
                if class_name == '10_yaozhed' or class_name == 'd':
                    print('xml_path ', xml_path, ' class_name ', class_name)

                # 更新统计信息
                self.total_boxes += 1
                self.stats[class_name]['count'] += 1
                self.stats[class_name]['widths'].append(width)
                self.stats[class_name]['heights'].append(height)
                self.stats[class_name]['ratios'].append(width / height)
        else:  # 输入yolo txt格式
            txt_path, img_width, img_height = info 
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id, x_center, y_center, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    self.stats[class_id]['count'] += 1
                    self.stats[class_id]['widths'].append(width)
                    self.stats[class_id]['heights'].append(height)
                    self.stats[class_id]['ratios'].append((width * img_width) / (height*img_height))
                    self.total_boxes += 1

        self.total_images += 1
        return

    def report(self, log_file=None):
        if log_file is not None:
            with open(log_file, 'a+') as f:
                f.write(f"Total positive images: {self.total_images}  total negative images: {self.total_background_images}\n")
                f.write(f"Total boxes: {self.total_boxes}\n")
                for class_name, stat in self.stats.items():
                    class_name_real = ''
                    if class_name in self.id2class:
                        class_name_real = self.id2class[class_name]
                    f.write(f"Class {class_name} {class_name_real}: \n")
                    f.write(f"\tCount: {stat['count']} \n")
                    f.write(f"\tWidth average: {sum(stat['widths']) / len(stat['widths'])}, min: {min(stat['widths'])}, max: {max(stat['widths'])}\n")
                    f.write(f"\tHeight average: {sum(stat['heights']) / len(stat['heights'])}, min: {min(stat['heights'])}, max: {max(stat['heights'])}\n")
                    f.write(f"\tAspect ratio (width / height) average: {sum(stat['ratios']) / len(stat['ratios'])}, min: {min(stat['ratios'])}, max: {max(stat['ratios'])}\n")

        print(f"Total positive images: {self.total_images}  total negative images: {self.total_background_images}")
        print(f"Total boxes: {self.total_boxes}")
        for class_name, stat in self.stats.items():
            class_name_real = ''
            if class_name in self.id2class:
                class_name_real = self.id2class[class_name]
            print(f"Class {class_name} {class_name_real}:")
            print(f"\tCount: {stat['count']}")
            print(f"\tWidth average: {sum(stat['widths']) / len(stat['widths'])}, min: {min(stat['widths'])}, max: {max(stat['widths'])}")
            print(f"\tHeight average: {sum(stat['heights']) / len(stat['heights'])}, min: {min(stat['heights'])}, max: {max(stat['heights'])}")
            print(f"\tAspect ratio (width / height) average: {sum(stat['ratios']) / len(stat['ratios'])}, min: {min(stat['ratios'])}, max: {max(stat['ratios'])}")


def convert_conr_to_yolo_format(x_min, y_min, x_max, y_max, image_width, image_height):
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    w = (x_max - x_min) / image_width
    h = (y_max - y_min) / image_height
    return x_center, y_center, w, h


# 同步调整图片大小和目标检测注释框的大小
def resize_image_and_annotations(img_path, xml_path, scale_percent, new_path=None, class2id = None):
    # 读取并调整图片大小
    img = cv2.imread(img_path)
    print('ori img ', img.shape)
    new_width = int(img.shape[1] * scale_percent / 100)
    new_height = int(img.shape[0] * scale_percent / 100) 
    img = cv2.resize(img, (new_width, new_height))

    ## 保存调整大小后的图片
    #cv2.imwrite(img_path, img)

    if xml_path is None:
        return img, None

    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if new_path is not None:
        # 修改xml中path信息
        root.find('path').text = new_path

    # 获取原始图片的大小
    size = root.find('size')
    old_width = int(size.find('width').text)
    old_height = int(size.find('height').text)
    print('old width ', old_width, ' new_width ', new_width)

    # 更新size子节点
    size.find('width').text = str(new_width)
    size.find('height').text = str(new_height) 

    # 遍历每个object标签
    for obj in root.findall('object'):
        # 获取并更新标注框的坐标
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text) * new_width / old_width
        ymin = int(bndbox.find('ymin').text) * new_height / old_height
        xmax = int(bndbox.find('xmax').text) * new_width / old_width
        ymax = int(bndbox.find('ymax').text) * new_height / old_height

        bndbox.find('xmin').text = str(int(xmin))
        bndbox.find('ymin').text = str(int(ymin))
        bndbox.find('xmax').text = str(int(xmax))
        bndbox.find('ymax').text = str(int(ymax))
        if class2id is not None:
            obj.find('name').text = class2id[obj.find('name').text]
        #print(' xmin ', xmin, ' ymin ', ymin, ' xmax ', xmax, ' ymax ', ymax, ' ', img.shape, ' resize ...')

    ## 保存更新后的XML文件
    save_path = xml_path.replace('.xml', '_resize.xml')
    tree.write(save_path)
    return img, save_path


# 显示图像和分割结果
def display_image_and_mask(image_path, mask_path):
    # 读取图像和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255

    overlay = np.zeros_like(image)
    overlay[mask == 1] = (0, 0, 255)  # BGR
    # 将覆盖层添加到原始图像上
    alpha = 0.5  # 设置透明度
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # 显示图像
    plt.imshow(image)
    plt.show()

def display_image_and_label(image_path, label_path):
    # 读取图像和标签
    img = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255

    # 创建一个新的figure
    fig, ax = plt.subplots(1, 2)

    # 显示图像和标签
    ax[0].imshow(img)
    ax[0].set_title('Image')
    ax[1].imshow(label)
    ax[1].set_title('Label')

    # 显示图像和标签
    plt.show()


# 根据目标检测xml、txt结果在图片上显示检测框，并显示对应类别
def display_annotations(img_path, xml_path, is_xml=True, id2class=None, title='', is_sub=False, class_colors=None, save_fig=False, save_path=None):
    # 读取图片
    print('image_path ', img_path)
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not is_sub:
        # 创建一个新的figure
        fig, ax = plt.subplots(figsize=(20,10))

        # 显示原始图片
        ax.imshow(img)

    if is_xml:  # voc xml
        # 解析XML文件
        print('xml_path ', xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 遍历每个object标签
        for obj in root.findall('object'):
            # 获取标注框的坐标
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            

            # 获取类别ID
            class_name = obj.find('name').text  # 需要根据你的类别名称映射到对应的ID
            if id2class is not None:
                class_name = id2class[class_name]
            color = class_colors.get(class_name, 'b')  # 如果类别未指定颜色，则默认为红色
            try:
                conf = float(obj.find('conf').text)
                if conf < 1:
                    class_name += (' '+str(round(conf, 2)))
            except:
                pass

            # 在图片上添加一个矩形框
            #print(' xmin ', xmin, ' ymin ', ymin, ' xmax ', xmax, ' ymax ', ymax, ' ', img.shape)
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor=color, facecolor='none')
            if is_sub:
                plt.gca().add_patch(rect)
            else:
                ax.add_patch(rect)

            # 在图片上添加类别名称
            plt.text(xmin-20, ymin-5, class_name, color='red', fontdict={'size':15})

    else:  # yolo txt
       # 读取txt文件
        print()
        with open(xml_path, 'r') as f:
            lines = f.readlines()

        # 遍历每一行
        for line in lines:
            # 解析YOLO格式的标注信息
            parts = line.strip().split()
            conf = 100
            if len(parts) == 6:
                class_id, conf, x_center, y_center, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            else:
                class_id, x_center, y_center, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # 计算VOC格式的坐标
            xmin = (x_center - width / 2) * w
            ymin = (y_center - height / 2) * h
            xmax = (x_center + width / 2) * w
            ymax = (y_center + height / 2) * h
            


            if id2class is not None:
                class_name = id2class[class_id]
            else:
                class_name = class_id
            color = class_colors.get(class_name, 'b')  # 如果类别未指定颜色，则默认为红色
            # 在图片上添加一个矩形框
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor=color, facecolor='none')
            if is_sub:
                plt.gca().add_patch(rect)
            else:
                ax.add_patch(rect)

            # 在图片上添加类别名称
            if conf < 1:
                class_name = class_name+' '+str(round(conf, 2))
            #plt.text(xmin, ymin, class_name, color='red', fontdict={'size':15})  # bbox=dict(facecolor='red', edgecolor='red')  color='white', 


    
    if is_sub:
        plt.imshow(img)
        plt.axis('off')
    else:
        plt.title(title)
        plt.axis('off')
        #plt.show()
    # 显示图片
    if save_fig:
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.savefig(xml_path.replace(xml_path.split('.')[-1], 'png'))
        plt.close(fig)


# 批量显示结果
def display_annotations_patches(image_paths, xml_paths=None, id2class=None):
    # 创建一个新的figure
    plt.figure(figsize=(20, 10))

    # 在两行中显示图片
    for i in range(len(image_paths)):
        plt.subplot((len(image_paths)-1)//5 + 1, 5, i + 1)
        if xml_paths is not None:
            display_annotations(image_paths[i], xml_paths[i], is_xml=True, id2class=id2class, is_sub=True)
        else:
            img_path = image_paths[i]
            # 读取图片
            if isinstance(img_path, str):
                img = cv2.imread(img_path)
            else:
                img = img_path
    
            h, w, _ = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.axis('off')


    # 显示figure
    plt.show()


# voc xml 转 yolo txt
def convert_voc_to_yolo(xml_path, img_width=None, img_height=None, class2id=None, is_save=True):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_annotations = []

    img_size = root.find('size')
    img_width = int(img_size.find('width').text)
    img_height = int(img_size.find('height').text)

    # 遍历每个object标签
    for obj in root.findall('object'):
        # 获取标注框的坐标
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 计算YOLO格式的中心点和宽高
        x_center = (xmin + xmax) / (2 * img_width)
        y_center = (ymin + ymax) / (2 * img_height)
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 获取类别ID
        class_id = obj.find('name').text  # 需要根据你的类别名称映射到对应的ID
        if class2id is not None:
            class_id = class2id[class_id]

        # 添加到结果列表
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    if is_save:
        # 保存为txt文件
        with open(xml_path.replace('.xml', '.txt'), 'w') as f:
            f.write("\n".join(yolo_annotations))
    return yolo_annotations


# yolo txt 转 voc xml
def convert_yolo_to_voc(txt_path, img_path, img_width, img_height, id2class = None, img_depth=3):
    # 创建XML文件的根节点
    root = ET.Element('annotation')

    # 添加folder, filename, path子节点
    ET.SubElement(root, 'folder').text = os.path.dirname(img_path)
    ET.SubElement(root, 'filename').text = os.path.basename(img_path)
    ET.SubElement(root, 'path').text = img_path

    # 添加source子节点
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    # 添加size子节点
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(img_width)
    ET.SubElement(size, 'height').text = str(img_height)
    ET.SubElement(size, 'depth').text = str(img_depth)

    # 添加segmented子节点
    ET.SubElement(root, 'segmented').text = '0'

    # 读取txt文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 遍历每一行
    for line in lines:
        # 解析YOLO格式的标注信息
        parts = line.strip().split()
        conf = 1
        if len(parts) > 5:
            class_id, conf, x_center, y_center, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        else:
            class_id, x_center, y_center, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # 计算VOC格式的坐标
        xmin = (x_center - width / 2) * img_width
        ymin = (y_center - height / 2) * img_height
        xmax = (x_center + width / 2) * img_width
        ymax = (y_center + height / 2) * img_height

        if id2class is not None and class_id in id2class:
            class_id = id2class[class_id]
        # 添加object子节点
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = class_id
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        ET.SubElement(obj, 'conf').text = str(conf)
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(xmin))
        ET.SubElement(bndbox, 'ymin').text = str(int(ymin))
        ET.SubElement(bndbox, 'xmax').text = str(int(xmax))
        ET.SubElement(bndbox, 'ymax').text = str(int(ymax))

    # 保存为XML文件
    tree = ET.ElementTree(root)
    tree.write(txt_path.replace('.txt', '.xml'))


def yolo_to_coco(yolo_txt_dir, output_coco_path, image_dir, is_pred=False, image2id = None):
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "spalling"},
            {"id": 2, "name": "rust"},
            {"id": 3, "name": "crack"},
            {"id": 4, "name": "discolor"},
            {"id": 5, "name": "blister"}] 
    }
    print('image_dir ', image_dir)
    annotation_id = 1
    id2image = {}
    image_id = 1
    if not os.path.isdir(yolo_txt_dir):
        files = [yolo_txt_dir]
        yolo_txt_dir = ''
    else:
        files = os.listdir(yolo_txt_dir)
    id2image = {}

    for filename in files:
        if filename.endswith('.txt'):
            id2image[image_id] = filename.split('/')[-1].split('.')[0]
            if is_pred:
                image_id = image2id[filename.split('/')[-1].split('.')[0]]
            image_name = filename.split('/')[-1].replace('.txt', '.jpg')
            image_path = os.path.join(image_dir, image_name)
            image_width, image_height = get_image_dimensions(image_path)

            coco_data["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": image_width,
                "height": image_height
            })
        else:
            continue

        with open(yolo_txt_dir+filename, 'r') as f:
            lines = f.readlines()
            #print(os.path.join(yolo_txt_dir, filename))
            for line in lines:
                #print(line)
                parts = line.strip().split()
                class_id = int(parts[0])
                extra = 0
                conf = 100
                if len(parts) == 6:
                    extra = 1
                    conf = float(parts[1])
                x_center = float(parts[1+extra])
                y_center = float(parts[2+extra])
                width = float(parts[3+extra])
                height = float(parts[4+extra])
                

                x_min, y_min, x_max, y_max = yolo_to_coco_bbox(x_center, y_center, width, height, image_width, image_height)

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id+1,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0,
                    'score': conf
                })

                annotation_id += 1
        image_id += 1

    with open(output_coco_path, "w") as coco_file:
        if is_pred:
            json.dump(coco_data["annotations"], coco_file)
        else:
            json.dump(coco_data, coco_file)
    return id2image

def yolo_to_coco_bbox(x_center, y_center, width, height, image_width, image_height):
    x_min = max(0, (x_center - width / 2) * image_width)
    y_min = max(0, (y_center - height / 2) * image_height)
    x_max = min(image_width, (x_center + width / 2) * image_width)
    y_max = min(image_height, (y_center + height / 2) * image_height)
    return x_min, y_min, x_max, y_max

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size


import xml.etree.ElementTree as ET
def voc_to_coco(voc_dir, coco_json_path, is_pred=False, image2id = None, prefix=''):
    data_dict = {}
    data_dict['images'] = []
    data_dict['annotations'] = []
    data_dict['categories'] = [{"id": 1, "name": "spalling"},
            {"id": 2, "name": "rust"},
            {"id": 3, "name": "crack"},
            {"id": 4, "name": "discolor"},
            {"id": 5, "name": "blister"}]  # 根据你的类别数量进行修改

    ann_id = 1
    if not os.path.isdir(voc_dir):
        files = [voc_dir]
        voc_dir = ''
    else:
        files = os.listdir(voc_dir)
    id2image = {}
    image_id = 1
    for filename in files:
        if filename.endswith(prefix+'.xml'):
            id2image[image_id] = filename.split('/')[-1].split('.')[0].replace(prefix,'')
            if is_pred:
                try:
                    image_id = image2id[filename.split('/')[-1].split('.')[0].replace(prefix,'')]
                except:
                    continue
            tree = ET.parse(voc_dir+ filename)
            root = tree.getroot()

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            data_dict['images'].append({
                'file_name': filename.split('/')[-1].replace(prefix,'').replace('.xml', '.jpg'),
                'height': height,
                'width': width,
                'id': image_id
            })

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = int(class_name)  # 需要根据你的类别名称映射到对应的ID
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                try:
                    conf = float(obj.find('conf').text)
                except:
                    conf = 100 
                data_dict['annotations'].append({
                    'area': (xmax - xmin) * (ymax - ymin),
                    'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'category_id': class_id+1,
                    'id': ann_id,
                    'image_id': image_id,
                    'iscrowd': 0,
                    'segmentation': [],
                    'ignore': 0,
                    'score': conf
                })
                ann_id += 1
        image_id += 1

    with open(coco_json_path, 'w') as f:
        if is_pred:
            json.dump(data_dict["annotations"], f)
        else:
            json.dump(data_dict, f)
    return id2image
