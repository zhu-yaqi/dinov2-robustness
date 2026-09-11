# DINOv2 视觉鲁棒性实验

本项目围绕预训练视觉模型在 **遮挡（Occlusion）与常见图像扰动** 下的鲁棒性展开实验，重点研究 DINOv2 等预训练视觉表征在输入发生变化时的稳定性，并尝试通过 **特征一致性约束（Consistency Regularization）** 提升模型对遮挡扰动的适应能力。

> **项目状态：Work in Progress / 实验进行中**

当前已完成实验框架、模型结构、扰动生成、训练流程、鲁棒性评估与结果可视化代码；完整实验结果仍在进一步运行与整理中。

---

# 研究问题

近年来，大规模预训练视觉模型能够学习较强的通用视觉表征。

但在实际场景中，输入图像可能受到：

* 局部遮挡
* 高斯噪声
* 模糊
* 亮度变化

等因素影响。

因此，本项目主要希望回答以下问题：

> **预训练视觉表征在图像受到扰动后是否仍然稳定？**

进一步地，本项目尝试研究：

> **能否通过约束原始图像与遮挡图像的特征一致性，提高模型对局部遮挡的鲁棒性？**

---

# 项目整体思路

基础分类流程为：

```text
Input Image
    ↓
Pretrained Backbone
    ↓
Feature Representation
    ↓
Classifier
    ↓
Prediction
```

项目对比多个 Backbone，包括：

* ResNet-50
* ConvNeXt V2 Tiny
* DINOv2 ViT-S/14

对于 DINOv2，主要采用：

```text
Input Image
    ↓
Frozen DINOv2 ViT-S/14
    ↓
Lightweight Adapter
    ↓
Classifier
```

在此基础上，引入遮挡一致性训练：

```text
Original Image ─────→ DINOv2 ─────→ Feature_original
                                         │
                                         │ Consistency Loss
                                         │
Occluded Image ─────→ DINOv2 ─────→ Feature_occluded
```

训练目标同时考虑：

```text
Classification Loss
+
λ × Consistency Loss
```

其中：

* Classification Loss 用于保证分类任务性能
* Consistency Loss 用于约束原图和扰动图的特征表示
* λ 用于控制一致性正则项强度

---

# 1. Baseline 对比

项目首先建立多个基础模型作为对照组。

目前包含：

```text
ResNet50
ConvNeXt V2 Tiny
DINOv2 ViT-S/14
```

目的是比较不同视觉模型在：

```text
Clean Images
vs.
Corrupted Images
```

上的性能变化。

重点关注：

* 干净测试集准确率
* 遮挡后准确率
* 高斯噪声下准确率
* 模糊情况下准确率
* 亮度变化情况下准确率

通过这些指标观察不同模型的鲁棒性差异。

---

# 2. DINOv2 特征提取

DINOv2 是一种基于自监督学习的大规模视觉预训练模型。

本项目主要使用：

```text
DINOv2 ViT-S/14
```

作为视觉 Backbone。

为了减少训练成本，并重点研究预训练表征本身的稳定性，实验中主要采用：

```text
Frozen Backbone
```

即：

> 不更新 DINOv2 主干网络参数。

训练过程中主要更新：

```text
Adapter
+
Classifier
```

这样能够：

* 降低显存与训练成本
* 减少对预训练模型的破坏
* 更直接地观察 DINOv2 原始视觉表征的鲁棒性

---

# 3. Lightweight Adapter

直接使用冻结的 Backbone 虽然能够保留预训练知识，但可能无法完全适应当前任务。

因此项目在 DINOv2 后加入轻量 Adapter：

```text
DINOv2 Feature
      ↓
Lightweight Adapter
      ↓
Task-specific Representation
      ↓
Classifier
```

Adapter 负责将预训练视觉特征映射到更适合当前任务的表示空间。

相比完整 Fine-tuning：

```text
Full Fine-tuning
```

这种设计需要训练的参数更少，也更适合快速实验和消融分析。

---

# 4. Occlusion Consistency Regularization

本项目尝试的核心方法是：

**Occlusion Consistency Regularization**

基本思想是：

同一张图像即使局部区域被遮挡，其高层语义通常不应该发生剧烈变化。

因此：

```text
Original Image
```

和：

```text
Occluded Image
```

经过 Backbone 后得到的特征应该保持一定程度的一致性。

---

## 训练流程

对于每张原始图像：

```text
x
```

生成一个遮挡后的版本：

```text
x_occ
```

分别计算：

```text
f(x)
```

和：

```text
f(x_occ)
```

然后使用一致性损失约束两者。

整体 Loss：

```text
L_total = L_cls + λ L_consistency
```

其中：

```text
L_cls
```

为分类损失，

```text
L_consistency
```

为原图与遮挡图之间的特征一致性损失。

---

# 5. 鲁棒性扰动设计

项目实现了多种输入扰动，用于测试模型对输入变化的稳定性。

---

## Occlusion

随机遮挡图像的局部区域。

用于模拟：

* 目标被其他物体遮挡
* 图像部分区域缺失
* 摄像头视野受到阻挡

---

## Gaussian Noise

向输入图像加入高斯噪声：

```text
x' = x + ε
```

用于观察模型对像素级随机扰动的敏感程度。

---

## Gaussian Blur

使用高斯模糊降低图像局部细节。

可以测试模型：

> 是否严重依赖高频纹理信息进行判断。

---

## Brightness Shift

对图像整体亮度进行调整。

用于模拟：

* 光照变化
* 曝光差异
* 不同环境采集条件

---

# 6. 鲁棒性评估

项目实现独立的 Robustness Evaluation 流程。

基本过程为：

```text
Trained Model
      ↓
Clean Test Set
      ↓
Baseline Accuracy

Trained Model
      ↓
Corrupted Test Set
      ↓
Robustness Accuracy
```

之后比较：

```text
Performance Drop
```

例如：

```text
Clean Accuracy - Occlusion Accuracy
```

用于衡量模型受到扰动后的性能下降。

相比只观察干净测试集准确率，这种方式能够更直接地分析模型的稳定性。

---

# 7. λ 消融实验

一致性约束的强度由参数：

```text
λ
```

控制。

如果 λ 太小：

> 一致性约束可能不足以对模型产生明显影响。

如果 λ 太大：

> 模型可能过度关注原图与扰动图的一致性，影响正常分类学习。

因此实验设计中包含：

```text
Different λ Values
```

进行消融比较。

主要希望分析：

```text
Classification Performance
        ↕
Robustness
```

之间的权衡关系。

---

# 实验设计

目前整体实验计划主要包括以下几个部分：

## Experiment 1：Baseline Comparison

比较：

```text
ResNet50
ConvNeXt V2 Tiny
DINOv2
```

在干净数据上的分类性能。

---

## Experiment 2：Robustness Evaluation

分别加入：

```text
Occlusion
Gaussian Noise
Gaussian Blur
Brightness Shift
```

比较各模型的性能下降。

---

## Experiment 3：Consistency Training

训练：

```text
Frozen DINOv2
+
Adapter
+
Occlusion Consistency Loss
```

并与普通 DINOv2 baseline 比较。

---

## Experiment 4：λ Ablation

改变：

```text
λ
```

分析一致性正则强度对：

```text
Clean Accuracy
Robustness Accuracy
Performance Drop
```

的影响。

---

# 项目结构

```text
.
├── README.md
├── requirements.txt
│
├── check_env.py
├── download_data.py
├── train_baseline.py
├── train_consistency.py
├── evaluate_robustness.py
├── plot_results.py
├── run_all.py
│
├── configs/
│   └── experiment_plan.md
│
├── scripts/
│   ├── 01_cpu_install.bat
│   ├── 01_cpu_install.sh
│   ├── 02_download_data.bat
│   ├── 03_quick_test.bat
│   └── 04_main_experiment.bat
│
├── src/
│   ├── __init__.py
│   ├── checkpoints.py
│   ├── constants.py
│   ├── corruptions.py
│   ├── data.py
│   ├── engine.py
│   ├── models.py
│   └── utils.py
│
└── tests/
    └── test_corruptions.py
```

---

# 核心模块

## `src/models.py`

主要负责：

* Backbone 构建
* DINOv2 加载
* Adapter
* Classifier
* Baseline 模型

---

## `src/corruptions.py`

负责实现鲁棒性测试使用的图像扰动：

```text
Occlusion
Gaussian Noise
Gaussian Blur
Brightness
```

---

## `src/data.py`

负责：

* 数据集加载
* 数据预处理
* DataLoader 构建

---

## `src/engine.py`

负责主要训练和评估流程。

包括：

* Training Loop
* Validation
* Loss 计算
* Metric 统计

---

## `train_baseline.py`

训练基础模型：

```text
ResNet50
ConvNeXt V2
DINOv2
```

用于后续鲁棒性比较。

---

## `train_consistency.py`

训练加入：

```text
Occlusion Consistency Regularization
```

的 DINOv2 模型。

---

## `evaluate_robustness.py`

对训练好的模型进行：

```text
Clean
Occlusion
Noise
Blur
Brightness
```

等场景下的统一评估。

---

## `plot_results.py`

用于生成实验结果图表，方便比较不同模型和不同扰动条件下的性能。

---

# 环境安装

推荐：

```text
Python 3.10+
```

首先克隆项目：

```bash
git clone https://github.com/zhu-yaqi/dinov2-robustness.git
cd dinov2-robustness
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

# 环境检查

可以首先运行：

```bash
python check_env.py
```

检查：

* Python
* PyTorch
* CUDA
* GPU
* 必要依赖

是否正常。

---

# 数据准备

运行：

```bash
python download_data.py
```

用于准备实验所需数据。

数据集本身不会提交到 GitHub 仓库。

---

# 训练 Baseline

运行：

```bash
python train_baseline.py
```

用于训练和评估基础视觉模型。

---

# 训练一致性模型

运行：

```bash
python train_consistency.py
```

训练：

```text
DINOv2
+
Adapter
+
Occlusion Consistency Regularization
```

---

# 鲁棒性评估

运行：

```bash
python evaluate_robustness.py
```

比较模型在不同图像扰动下的表现。

---

# 结果可视化

运行：

```bash
python plot_results.py
```

用于生成实验结果图。

---

# 完整实验

也可以使用：

```bash
python run_all.py
```

统一运行实验流程。

具体实验设置可以参考：

```text
configs/experiment_plan.md
```

---

# 当前进展

目前已完成：

* [x] 项目基础框架
* [x] 数据加载与预处理模块
* [x] ResNet50 Baseline
* [x] ConvNeXt V2 Baseline
* [x] DINOv2 Backbone
* [x] Lightweight Adapter
* [x] Occlusion 扰动
* [x] Gaussian Noise
* [x] Gaussian Blur
* [x] Brightness Shift
* [x] Consistency Loss 训练流程
* [x] 鲁棒性评估脚本
* [x] 结果绘图模块
* [x] 基础单元测试
* [ ] 完整 Baseline 实验
* [ ] Consistency 实验
* [ ] λ Ablation
* [ ] 最终实验结果汇总

---

# 当前关注的问题

项目目前更关注实验过程中的以下问题。

---

## 1. 预训练模型是否天然具有更强鲁棒性？

更强的预训练表示并不一定意味着：

```text
更高 Clean Accuracy
=
更高 Robustness
```

因此需要分别评估。

---

## 2. 模型到底依赖语义还是局部纹理？

如果遮挡、模糊或噪声会导致性能明显下降，说明模型仍然可能高度依赖：

* 局部纹理
* 高频信息
* 特定图像区域

而不是完全依赖更加稳定的全局语义。

---

## 3. Consistency Regularization 是否会产生 Trade-off？

增强：

```text
Robustness
```

可能同时影响：

```text
Clean Accuracy
```

因此需要通过 λ Ablation 分析：

```text
Accuracy ↔ Robustness
```

之间是否存在明显权衡。

---

# 项目局限

目前项目仍处于实验阶段，因此存在一些明确局限：

* 尚未完成所有实验组合
* 目前还没有完整的统计显著性分析
* λ 等超参数仍需要进一步调优
* 鲁棒性扰动主要是人工构造的图像变化
* 尚未覆盖更多真实分布偏移场景
* 暂未进行大规模 Fine-tuning 对比
* 当前重点是实验框架和方法验证，而不是追求 SOTA

因此目前不对：

> “Consistency Regularization 一定能够显著提升 DINOv2 鲁棒性”

做提前结论。

最终结论将根据完整实验结果进一步补充。

---

# 后续计划

下一阶段计划包括：

* 完成全部 Backbone Baseline
* 运行完整 Corruption Evaluation
* 完成 Consistency Regularization 实验
* 完成 λ Ablation
* 对比不同模型的 Robustness Drop
* 增加结果表格和可视化
* 分析 Clean Accuracy 与 Robustness 的关系
* 尝试不同 Occlusion Ratio
* 探索更复杂的 Feature Consistency 方法
* 进一步分析 DINOv2 表征在扰动前后的特征变化

---

# 技术栈

```text
Python
PyTorch
Torchvision
DINOv2
Vision Transformer
ResNet
ConvNeXt
Matplotlib
```

涉及的主要方向：

```text
Computer Vision
Self-Supervised Learning
Pretrained Visual Representation
Vision Transformer
Robustness Evaluation
Image Corruption
Consistency Regularization
Transfer Learning
```

---

# 说明

本项目当前主要用于学习和探索预训练视觉模型的鲁棒性问题。

随着实验继续进行，后续将逐步补充：

* 完整实验结果
* 对比表格
* 鲁棒性曲线
* 消融实验结果
* 实验分析与结论
