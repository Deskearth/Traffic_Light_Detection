# 参考：https://aistudio.baidu.com/aistudio/projectdetail/442375

# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

# 图像预处理
from paddlex.det import transforms
# 训练集图像预处理
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])
# 评估集图像预处理
eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])

# 按VOC数据集格式加载数据
train_dataset = pdx.datasets.VOCDetection(
    data_dir='./dataset',
    file_list='./dataset/train_list.txt',
    label_list='./dataset/label_list.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='./dataset',
    file_list='./dataset/val_list.txt',
    label_list='./dataset/label_list.txt',
    transforms=eval_transforms)


num_classes = len(train_dataset.labels)
# 引入YOLOv3模型
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
# 开始训练
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_interval_epochs=20,
    save_dir='output/yolov3_darknet53')

# 测试及结果可视化
image_name = './测试集/red_1032.jpg'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.2, save_dir='./output/yolov3_mobilenetv1')