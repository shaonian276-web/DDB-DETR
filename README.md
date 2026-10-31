# DDB-DETR

**Density-Guided Frequency Modulation and Detail–Background Balancing for Tiny Object Detection**

面向微小目标检测的密度引导频率调制与细节—背景平衡

## 方法简介

DDB-DETR 基于 RT-DETRv2 和 PResNet-34，针对航拍图像中微弱目标细节与复杂背景纹理难以区分的问题，在混合编码器之前依次引入三个模块：

- **DCEM**：保留局部细节并聚合上下文。
- **DGFQ**：利用学习得到的密度响应调制空间高通残差，无需额外密度标注。
- **BSR**：通过前景增强和加权背景分支减法进行特征重校准。

三个特征层独立执行 **DCEM → DGFQ → BSR**，保留端到端预测接口。DGFQ 在空间域中工作，不使用 FFT，也不改变查询数量。

## 主要结果

两个数据集均在验证集上评估，输入尺寸为 **640 × 640**。

| 数据集 | 方法 | AP | AP50 | AP75 |
| :--- | :--- | ---: | ---: | ---: |
| VisDrone | RT-DETRv2 | 28.1 | 48.1 | 28.2 |
| VisDrone | **DDB-DETR** | **31.2** | **50.2** | **30.4** |
| AI-TOD-v2 | RT-DETRv2 | 28.5 | 64.1 | 22.3 |
| AI-TOD-v2 | **DDB-DETR** | **31.8** | **67.2** | **25.8** |

相较基线，VisDrone 和 AI-TOD-v2 的 AP 分别提升 **3.1** 和 **3.3** 个百分点；VisDrone 的小目标 APs 从 **19.4** 提升至 **21.5**。

*结果依据 2026-10-31 修订版论文。AP 与 AP50 为四次运行均值，其余指标为点估计。模型参数量为 40.0M，计算量为 285 GFLOPs；尚未报告延迟或吞吐量。*

## 数据集下载

| 数据集 | 官方下载入口 |
| :--- | :--- |
| **VisDrone2019-DET** | [官方下载页：训练集、验证集及测试集（百度网盘 / Google Drive）](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images) |
| **AI-TOD-v2** | [官方下载（Google Drive）](https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing) · [官方项目与说明](https://github.com/Chasel-Tsui/mmdet-aitod) |

请使用 AI-TOD-v2 的更新标注，并遵循各数据集的使用及引用要求。

## 代码与作者

代码位于 [`rtdetrv2_pytorch/`](rtdetrv2_pytorch/)，包含模型实现、配置文件和训练入口。

**Qizheng Liu · Chunyan Wang**  
辽宁工程技术大学，智能科学与工程学院  
联系邮箱：shaonian276@gmail.com
