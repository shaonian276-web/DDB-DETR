# DDB-DETR

**Density-Guided Frequency Modulation and Detail–Background Balancing for Tiny Object Detection**

面向微小目标检测的密度引导频率调制与细节—背景平衡

## 方法简介

DDB-DETR 基于 RT-DETRv2 和 PResNet-34，针对航拍图像中微弱目标细节与复杂背景纹理难以区分的问题，在混合编码器之前依次引入三个模块：

- **DCEM**：保留局部细节并聚合上下文。
- **DGFQ**：利用学习得到的密度响应调制空间高通残差，无需额外密度标注。
- **BSR**：通过前景增强和加权背景分支减法进行特征重校准。

三个特征层独立执行 **DCEM → DGFQ → BSR**，保留端到端预测接口。
## 数据集下载

| 数据集 | 官方下载入口 |
| :--- | :--- |
| **VisDrone2019-DET** | [官方下载页：训练集、验证集及测试集（百度网盘 / Google Drive）](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images) |
| **AI-TOD-v2** | [官方下载（Google Drive）](https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing) · [官方项目与说明](https://github.com/Chasel-Tsui/mmdet-aitod) |

请使用 AI-TOD-v2 的更新标注，并遵循各数据集的使用及引用要求。

