# Traffic_Light_Detection
Traffic Light Detection

## [PaddleX版本](https://github.com/Deskearth/Traffic_Light_Detection/tree/master/PaddleX/TrafficLight)
位于PaddleX目录下
### 数据集
数据集下载：
* 训练集：https://aistudio.baidu.com/aistudio/datasetdetail/30989
* 测试集：https://aistudio.baidu.com/aistudio/datasetdetail/30990

### 框架
PaddlePaddle

### 注意事项
1. 将训练集和测试集数据解压到与py文件同一目录下
2. 需要手动按照VOC数据格式创建文件夹，VOC数据格式参考：https://paddlex.readthedocs.io/zh_CN/latest/datasets.html
3. 手动创建完文件夹之后运行data_process.py文件来将数据集转化为VOC数据集格式
4. 建议在百度的GPU云主机上运行因为
    * 在本地GPU运行大概要3个小时
    * 本地库依赖不太好安装
5. 如果要在本地运行，需要安装
    * paddlex
    * cython
    * pycocotools

### 上述库安装方法
  请参考paddlex文档页：https://paddlex.readthedocs.io/zh_CN/latest/install.html  
  但是pycocotools无法按照文档页中的方法安装，请参考：https://www.jianshu.com/p/8658cda3d553 的方案2

## [PyTorch版本](https://github.com/Deskearth/Traffic_Light_Detection/tree/master/torch)
### 数据集
数据集下载：[https://aistudio.baidu.com/aistudio/datasetdetail/34356](https://aistudio.baidu.com/aistudio/datasetdetail/34356)

### 框架
PyTorch
