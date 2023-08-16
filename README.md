# 📘Zoomir使用说明

## 🛠️Step 1：环境配置

......

## 📄Step 2：数据集准备

👀数据集目录格式如下：

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
│   │   ├── PNGImages_2x(api生成的)  
│   │   ├── PNGImages_2x_bicubic  
│   │   ├── PNGImages_2x_edsr  
│   │   ├── PNGImages_2x_srcnn  
│   │   ├── PNGImages_2x_srgan  
│   │   ├── PNGImages_2x_swinir  
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
│   │   ├── trainvaltest.txt(for mmagic)  
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
│   │   ├── IRSTD1k_Img_2x_edsr  
│   │   ├── IRSTD1k_Img_2x_srcnn  
│   │   ├── IRSTD1k_Img_2x_srgan  
│   │   ├── IRSTD1k_Img_2x_swinir  
│   │   ├── IRSTD1k_Label(not use)  
│   │   ├── Splits(for mmdet)  
│   │   │   ├── test.txt  
│   │   │   ├── trainval.txt  
│   │   │   ├── trainvaltest.txt  
│   │   ├── test.txt(for mmagic)  
│   │   ├── trainval.txt(for mmagic)
│   │   ├── trainvaltest.txt(for mmagic)  
```

## 🚀Step 3：模型训练

### 📊训练红外超分模型

#### ✨CPU训练：

```
CUDA_VISIBLE_DEVICES=-1 python tools/train_mmagic.py [model_config]
```

具体示例如下：

```
# train srcnn
CUDA_VISIBLE_DEVICES=-1 python tools/train_mmagic.py configs/srcnn/srcnn_x2k915_1xb16-1000k_irstd.py
# train srgan
CUDA_VISIBLE_DEVICES=-1 python tools/train_mmagic.py configs/srgan_resnet/srgan_x2c64b16_1xb16-100k_irstd.py
```

#### ✨✨单GPU训练：

```
python tools/train_mmagic.py [model_config]
```

具体示例如下：

```
# train srcnn
python tools/train_mmagic.py configs/srcnn/srcnn_x2k915_1xb16-1000k_irstd.py
# train srgan
python tools/train_mmagic.py configs/srgan_resnet/srgan_x2c64b16_1xb16-100k_irstd.py
```

### 📊训练红外小目标检测模型

#### ✨CPU训练：

```
CUDA_VISIBLE_DEVICES=-1 python tools/train_det.py [model_config]
```

具体示例如下：

```
# train fcos
CUDA_VISIBLE_DEVICES=-1 python tools/train_det.py configs/fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_voc.py
```

#### ✨✨单GPU训练：

```
python tools/train_det.py [model_config]
```

具体示例如下：

```
# train fcos
python tools/train_det.py configs/fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_voc.py
```

## 🚀Step 4：模型测试

### 📊测试红外超分模型

#### ✨CPU测试：

```
CUDA_VISIBLE_DEVICES=-1 python tools/test_mmagic.py [model_config] [checkpoint]
```

具体示例如下：

```
# test srcnn
CUDA_VISIBLE_DEVICES=-1 python tools/test_mmagic.py configs/srcnn/srcnn_x2k915_1xb16-1000k_irstd.py work_dirs\srcnn_x2k915_1xb16-1000k_irstd\iter_500.pth
```

#### ✨✨单GPU测试：

```
python tools/test_mmagic.py [model_config] [checkpoint]
```

具体示例如下：

```
# test srcnn
CUDA_VISIBLE_DEVICES=-1 python tools/test_mmagic.py configs/srcnn/srcnn_x2k915_1xb16-1000k_irstd.py work_dirs\srcnn_x2k915_1xb16-1000k_irstd\iter_500.pth
```

### 📊训练红外小目标检测模型

#### ✨CPU测试：

```
CUDA_VISIBLE_DEVICES=-1 python tools/test_det.py [model_config] [checkpoint]
```

具体示例如下：

```
# train fcos
CUDA_VISIBLE_DEVICES=-1 python tools/test_det.py configs/fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_voc.py work_dirs/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_voc/best_pascal_voc_mAP_epoch_1.pth
```

#### ✨✨单GPU测试：

```
python tools/test_det.py [model_config] [checkpoint]
```

具体示例如下：

```
# train fcos
python tools/test_det.py configs/fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_voc.py work_dirs/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_voc/best_pascal_voc_mAP_epoch_1.pth
```
