# 功能：将数据转换为符合VOC数据集格式
# VOC数据集格式说明：https://paddlex.readthedocs.io/zh_CN/latest/datasets.html
# 提示：请提前手动创建VOC数据集格式要求的文件夹

import os
import random
import shutil

raw_train_file_dir = './训练集'
raw_test_file_dir = './测试集'
VOC_jpg_dir = './dataset/JPEGImages'
VOC_xml_dir = './dataset/Annotations'
# 训练集大小
train_len = 1600
# 验证集大小
val_len = 2000-train_len


raw_train_file_list = os.listdir(raw_train_file_dir)

# 搬移训练集数据
for i, file_name in enumerate(raw_train_file_list):
    if os.path.splitext(file_name)[1] == '.jpg':
        shutil.copy((raw_train_file_dir+'/'+file_name), VOC_jpg_dir)
    elif os.path.splitext(file_name)[1] == '.xml':
        shutil.copy((raw_train_file_dir+'/'+file_name), VOC_xml_dir)
    else:
        print('文件复制过程出现异常')

print('文件搬移完成')

# 创建列表文件train_list.txt、val_list.txt、label_list.txt
VOC_jpg_list = os.listdir(VOC_jpg_dir)
random.shuffle(VOC_jpg_list)#随机打乱

train_VOC_list = []  # 保存训练集图片的文件名
val_VOC_list = []  # 保存评估集图片的文件名
for i, file_name in enumerate(VOC_jpg_list):
    if i < train_len:
        train_VOC_list.append(file_name)
    else:
        val_VOC_list.append(file_name)

train_list_txt = open('./dataset/train_list.txt', "w")
val_list_txt = open('./dataset/val_list.txt', "w")
label_list_txt = open('./dataset/label_list.txt', "w")

# 写入train_list.txt
for i, jpg_name in enumerate(train_VOC_list):
    xml_name = os.path.splitext(jpg_name)[0] + '.xml'
    train_list_txt.write('JPEGImages/' + jpg_name + ' ' + 'Annotations/' + xml_name + '\n')

# 写入val_list.txt
for i, jpg_name in enumerate(val_VOC_list):
    xml_name = os.path.splitext(jpg_name)[0] + '.xml'
    val_list_txt.write('JPEGImages/' + jpg_name + ' ' + 'Annotations/' + xml_name + '\n')

# 写入label_list.txt
label_list_txt.write('green\n')
label_list_txt.write('red')