import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy
import torch.nn as nn

item_table = ['绿灯','红灯']

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./trafficdata"
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 32
# Number of epochs to train for
num_epochs = 1
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

input_size = 224

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize([input_size,input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize([input_size,input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
        batch_size=batch_size, shuffle=True) for x in ["train", "val"]}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################
test_model = models.resnet18(pretrained=False)
num_ftrs = test_model.fc.in_features
test_model.fc = nn.Linear(num_ftrs, num_classes)
test_model.load_state_dict(torch.load('./traffic_model.pt'))
test_model.eval()#这个特别重要！
#####################################

# ######################结果可视化###############################
# test_num = 250
# img = dataloaders_dict['val'].dataset[test_num][0]
# tag = dataloaders_dict['val'].dataset[test_num][1]
# input = img.unsqueeze(0)
# print('标签:', item_table[tag])
# prediction = test_model(input).argmax(dim=1)
# print('预测结果:',item_table[prediction.item()])
# img=torchvision.utils.make_grid(img)
# plt.imshow(np.transpose(img,(1,2,0)))
# plt.show()
# #############################################################

########################准确率测试#############################
cnt = 0.
with torch.no_grad():
    for i in range(399):
        test_num = i
        img = dataloaders_dict['val'].dataset[test_num][0]
        tag = dataloaders_dict['val'].dataset[test_num][1]
        input = img.unsqueeze(0)
        print('tag:', tag)
        prediction = test_model(input).argmax(dim=1)
        print('prediction:',prediction.item())
        cnt += (prediction == tag)
print(cnt.item()/400.)
#############################################################
