<<<<<<< HEAD
# Torpedo Rotate

## 鱼雷哥操作手册

### Step 1：环境安装

1. `pip install openmim`
2. `mim install mmcv-full`
3. `mim install mmdet`
4. `pip install -r requirements/build.txt`
5. `pip install -v -e .`

其中 `mmcv-full==1.7.1`, `mmdet==2.28.1`, 可进行离线安装该版本。离线安装完之后执行 `pip install -v -e .` 命令。

### Step 2: 数据集准备

数据集目录格式如下：

```shell
.
├── configs
├── data
│   ├── radar
│   │   ├── test
│   │   │   ├── annfiles
│   │   │   │   ├── NRxVideoRecord_00001_cyc8_377355_377473.txt
│   │   │   │   └── ······
│   │   │   └── images
│   │   │       ├── NRxVideoRecord_00001_cyc8_377355_377473.png
│   │   │       └── ······
│   │   └── trainval
│   │       ├── annfiles
│   │       │   ├── NRxVideoRecord_00000_cyc2_376641_376760.txt
│   │       │   └── ······
│   │       └── images
│   │           ├── NRxVideoRecord_00000_cyc2_376641_376760.png
│   │           └── ······
│   └── ······
├── tools
└── ······
```

请按照上面格式来存放数据集，数据集准备完成之后，确保 `configs/base/datasets/dotav1.py` 里面 `data_root` 和数据集目录保持一致。
可以使用一下命令来进行数据集自动划分：

```shell
python tools/split.py [ori_path] [new_path]
```

其中，`ori_path` 表示需要划分的数据集，里面应该由一个 `png` 文件夹和一个 `xml` 文件夹组成，`new_path` 表示划分之后的数据集。
具体示例如下：

```shell
python tools/split.py data/ori_img data/split_data
```

需要注意的是：

- 以本例子为例，对于 `data/split_data` 来说，在划分数据集之前，`data` 文件夹要手动创建，否则会报错，`split_data` 文件夹不需要手动创建。
- 为了防止误删除数据，如果 `data` 目录下已经存在 `split_data`，我们将抛出异常并提示该数据集已存在。
- 可以在 `split.py` 中 `parse_args()` 部分修改 `trainval_seq` 和 `test_seq`，自行进行序列划分。
- `split.py`中包含了 `voc2dota` 操作，如果你使用该划分脚本，则不需要在进行 `voc2dota` 操作。

由于训练数据的标注格式为 dota 类型，所以需要执行 `voc2dota.py` 将 xml 转为 txt，如下：

```shell
python tools/voc2dota.py [xml_path] [txt_path]
```

具体示例如下：

```shell
python tools/voc2dota.py data/radar/trainval/xml data/radar/trainval/annfiles
```

### Step 3：模型训练

#### 图片分辨率问题
对于RepPoints方法，如果GPU显存为24G，所有数据集均可使用2048 x 5000的分辨率进行训练测试（Batch Size设置为2）。

#### 单卡训练

```shell
python tools/train.py [config]
```

具体示例如下：

```shell
'''FCOS'''
python tools/train.py configs/rotated_fcos/rotated_fcos_sep_angle_r34_fpn_1x_dota_le90.py

'''FCOS + Dual Attention'''
python tools/train.py configs/rotated_fcos/rotated_fcos_sep_angle_da_r34_fpn_1x_dota_le90.py

'''FCOS + DyReLU'''
python tools/train.py configs/rotated_fcos/rotated_fcos_sep_angle_dyrelu_r34_fpn_1x_dota_le90.py

'''RepPoints (le90)'''
python tools/train.py configs/oriented_reppoints/oriented_reppoints_r34_fpn_1x_dota_le135.py

'''RepPoints + Dual Attention (le90) (性能下降)'''
python tools/train.py configs/oriented_reppoints/oriented_reppoints_da_r34_fpn_1x_dota_le135.py
```


#### 多卡训练

```shell
CUDA_VISIBLE_DEVICES=[GPU ids] tools/dist_train.sh [config] [GPU numbers]
```

具体示例如下：

```shell
'''FCOS'''
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/rotated_fcos/rotated_fcos_sep_angle_r34_fpn_1x_dota_le90.py 4

'''FCOS + Dual Attention'''
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/rotated_fcos/rotated_fcos_sep_angle_da_r34_fpn_1x_dota_le90.py 4

'''FCOS + DyReLU'''
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/rotated_fcos/rotated_fcos_sep_angle_dyrelu_r34_fpn_1x_dota_le90.py 4

'''RepPoints'''
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/oriented_reppoints/oriented_reppoints_r34_fpn_1x_dota_le135.py 4

'''RepPoints + Dual Attention (性能下降)'''
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/oriented_reppoints/oriented_reppoints_da_r34_fpn_1x_dota_le135.py 4
```

如果是第一次执行该命令，需要首先执行一下命令给 `./tools/dist_train.sh` 脚本赋予可执行权限：

```shell
chmod 777 ./tools/dist_train.sh
```

### Step 4：模型测试

#### 测试输出可视化结果

```python
python tools/test.py [config] [checkpoint] [--eval mAP] [--show-dir] [--show-score-thr]
```

其中，`--show-score-thr` 为可视化结果的检测阈值，默认为 0.05（可不选）。

具体示例如下：

```python
'''FCOS'''
python tools/test.py configs/rotated_fcos/rotated_fcos_sep_angle_r34_fpn_1x_dota_le90.py \
    work_dirs/rotated_fcos/rotated_fcos_sep_angle_r34_fpn_1x_dota_le90/best_mAP_epoch_1.pth \
    --eval mAP --show-dir visual/ --show-score-thr 0.05

'''FCOS + Dual Attention'''
python tools/test.py configs/rotated_fcos/rotated_fcos_sep_angle_da_r34_fpn_1x_dota_le90.py \
    work_dirs/rotated_fcos/rotated_fcos_sep_angle_da_r34_fpn_1x_dota_le90/best_mAP_epoch_1.pth \
    --eval mAP --show-dir visual/ --show-score-thr 0.05

'''FCOS + DyReLU'''
python tools/test.py configs/rotated_fcos/rotated_fcos_sep_angle_dyrelu_r34_fpn_1x_dota_le90.py \
    work_dirs/rotated_fcos/rotated_fcos_sep_angle_dyrelu_r34_fpn_1x_dota_le90/best_mAP_epoch_1.pth \
    --eval mAP --show-dir visual/ --show-score-thr 0.05

'''RepPoints'''
python tools/test.py configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
    work_dirs/oriented_reppoints/oriented_reppoints_r34_fpn_1x_dota_le135/best_mAP_epoch_1.pth \
    --eval mAP --show-dir visual/ --show-score-thr 0.05

'''RepPoints + Dual Attention (性能下降)'''
python tools/test.py configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
    work_dirs/oriented_reppoints/oriented_reppoints_da_r34_fpn_1x_dota_le135/best_mAP_epoch_1.pth \
    --eval mAP --show-dir visual/ --show-score-thr 0.05
```




#### 测试输出 xml 结果

范式：

```python
python tools/get_xml.py [config] [checkpoint] [--xml-dir] [--score-thr]
```

其中 `--score-thr` 为输出 xml 的检测阈值，默认为 0.05（可不选）。

具体示例：

```python
'''FCOS'''
python tools/get_xml.py configs/rotated_fcos/rotated_fcos_sep_angle_r34_fpn_1x_dota_le90.py \
    work_dirs/rotated_fcos/rotated_fcos_sep_angle_r34_fpn_1x_dota_le90/best_mAP_epoch_1.pth \
     --xml-dir xml/ --score-thr 0.05

'''FCOS + Dual Attention'''
python tools/get_xml.py configs/rotated_fcos/rotated_fcos_sep_angle_da_r34_fpn_1x_dota_le90.py \
    work_dirs/rotated_fcos/rotated_fcos_sep_angle_da_r34_fpn_1x_dota_le90/best_mAP_epoch_1.pth \
     --xml-dir xml/ --score-thr 0.05

'''FCOS + DyReLU'''
python tools/get_xml.py configs/rotated_fcos/rotated_fcos_sep_angle_dyrelu_r34_fpn_1x_dota_le90.py \
    work_dirs/rotated_fcos/rotated_fcos_sep_angle_dyrelu_r34_fpn_1x_dota_le90/best_mAP_epoch_1.pth \
     --xml-dir xml/ --score-thr 0.05

'''RepPoints'''
python tools/get_xml.py configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
    work_dirs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135/best_mAP_epoch_1.pth \
     --xml-dir xml/ --score-thr 0.05

'''RepPoints + Dual Attention (性能下降)'''
python tools/get_xml.py configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
    work_dirs/oriented_reppoints/oriented_reppoints_da_r34_fpn_1x_dota_le135/best_mAP_epoch_1.pth \
     --xml-dir xml/ --score-thr 0.05
```

## Results
|          **方法**           | **第一批数据** | **第二批数据** |
|:---------------------:|:---------:|:---------:|
|         FCOS          |   75.26   |   70.34   |
|   FCOS + Attention    |   78.34   |   70.52   |
|       RepPoints       |   80.25   |   74.65   |
| RepPoints + Attention |           |           |

## Package Dependency

具体请见 `requirements/install.txt`

## Pretrained Models

训练时会加载骨干网络的预训练模型，离线状态下由于网络不通会报错 `URLError`：`urllib.error.URLError: <urlopen error [Errno -2] Name or service not knowns>`。
解决办法如下：

1. 自行下载然后存放到本地文件夹，比如 `checkpoints`；
2. 修改 backbone 的 init_cfg 中的 checkpoint 路径，比如 `checkpoint='checkpoints/resnet50_caffe.pth'`

下面是一些常用的 backbone 的预训练权重下载地址^[[SOURCE CODE FOR TORCHVISION.MODELS.RESNET](https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html)]：

<!-- - ResNet50-Caffe：<https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth>
- ResNet50：<https://download.pytorch.org/models/resnet50-19c8e357.pth>
- ResNet34：<https://download.pytorch.org/models/resnet34-333f7ec4.pth> -->

```python
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
```

## Codebase

- [MMRotate V0.3.4](https://github.com/open-mmlab/mmrotate/tree/main)

## 参考资料

=======
# zoomir
## The data folder structure:
```shell
zoomir  
├── zoomir  
├── tools  
├── configs  
├── data  
│   ├── SIRSTdevkit-master  
│   │   ├── PNGImages  
│   │   │   ├── 211_HD_1.png  
│   │   │   ├── 211_HD_126.png  
│   │   │   ├── ...  
│   │   │   ├── TH0517_12.png  
│   │   ├── PNGImages_2x(sr)  
│   │   ├── PNGImages_2x_bicubic  
│   │   ├── SIRST   
│   │   │   ├── BBox  
│   │   │   │   ├── 211_HD_1.xml  
│   │   │   │   ├── 211_HD_126.xml  
│   │   │   │   ├── ...   
│   │   │   │   ├── TH0517_12.xml  
│   │   │   ├── BBox_2X  
│   │   │   ├── BinaryMask(not use)  
│   │   │   ├── PaletteMask(not use)  
│   │   ├── SkySeg(not use)  
│   │   ├── Splits(for mmdet)  
│   │   │   ├── test.txt  
│   │   │   ├── trainval.txt  
│   │   ├── test.txt(for mmagic)  
│   │   ├── trainval.txt(for mmagic)  
│   ├── IRSTD-1k  
│   │   ├── Annotations  
│   │   │   ├── XDU0.xml  
│   │   │   ├── XDU0.xml  
│   │   │   ├── ...  
│   │   │   ├── XDU1000.xml  
│   │   ├── Annotations_2x  
│   │   ├── IRSTD1k_Img  
│   │   │   ├── XDU0.png  
│   │   │   ├── XDU1.png  
│   │   │   ├── ...  
│   │   │   ├── XDU1000.png  
│   │   ├── IRSTD1k_Img_2x(sr)  
│   │   ├── IRSTD1k_Img_2x_bicubic  
│   │   ├── IRSTD1k_Label(not use)  
│   │   ├── Splits(for mmdet)  
│   │   │   ├── test.txt  
│   │   │   ├── trainval.txt  
│   │   │   ├── trainvaltest.txt  
│   │   ├── test.txt(for mmagic)  
│   │   ├── trainval.txt(for mmagic)  
```
>>>>>>> 6caf377c6e46a8bf31d1be1caf6810a953db19e7
