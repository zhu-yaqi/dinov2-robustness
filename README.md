# 基于 DINOv2 的遮挡一致性视觉鲁棒性研究

本项目研究预训练视觉模型在 **遮挡、噪声、模糊和亮度变化** 等扰动条件下的鲁棒性，并基于 **DINOv2 ViT-S/14** 设计了一种轻量的 **Occlusion-Consistency Adapter**。

核心思路是：

> 在冻结 DINOv2 主干网络的情况下，仅训练一个轻量 Adapter 和分类器，同时约束同一张图像的干净版本与遮挡版本在特征空间中保持一致。

项目在 CIFAR-10 上与 ResNet50、ConvNeXt V2 Tiny 和原始 DINOv2 进行统一对比，并进一步通过不同一致性权重 λ 的消融实验分析性能提升来源。

---

## 项目亮点

* 使用 **DINOv2 ViT-S/14** 作为冻结视觉 Backbone
* 设计轻量 `384 → 256` Adapter，仅训练约 **10.16 万参数**
* 引入 **Occlusion Consistency Loss**，约束原图与遮挡图特征稳定性
* 对比 ResNet50、ConvNeXt V2 Tiny、DINOv2 等视觉模型
* 构建 **4 类扰动 × 5 个强度 = 20 个鲁棒性测试条件**
* 完成 λ = 0 / 0.1 / 0.5 / 1.0 一致性权重消融实验
* 分析 Clean Accuracy 与 Robustness 之间的关系
* 保留未符合预期的实验结果，并讨论方法适用范围和局限

---

# 核心实验结果

## 主实验

| 模型                |     Clean | Occlusion |     Noise |      Blur | Brightness | Corruption Avg. |
| ----------------- | --------: | --------: | --------: | --------: | ---------: | --------------: |
| ResNet50          |     92.31 |     77.45 |     39.44 |     90.92 |      90.73 |           74.64 |
| ConvNeXt V2 Tiny  |     94.47 |     83.70 |     68.64 |     92.46 |      89.05 |           83.46 |
| DINOv2 ViT-S/14   |     97.45 |     82.66 |     70.91 |     97.56 |      93.83 |           86.24 |
| **DINOv2 + Ours** | **97.67** | **85.05** | **72.58** | **97.75** |  **94.78** |       **87.54** |

相比原始 DINOv2：

```text
Clean Accuracy
97.45% → 97.67%
+0.22 percentage points
```

```text
Mean Corruption Accuracy
86.24% → 87.54%
+1.30 percentage points
```

```text
Mean Occlusion Accuracy
82.66% → 85.05%
+2.40 percentage points
```

在保持干净图像性能基本不下降的情况下，模型对扰动输入的稳定性得到一定改善。

---

# 1. 研究问题

人类在物体部分被遮挡、图像变模糊或光照发生变化时，通常仍然能够保持较稳定的对象识别能力。

相比之下，视觉模型即使在标准数据集上准确率很高，也可能在输入受到扰动后出现明显性能下降。

因此，本项目主要关注三个问题：

### Question 1

DINOv2 与 ResNet50、ConvNeXt V2 Tiny 相比，在不同视觉扰动下有什么优势和不足？

### Question 2

能否通过轻量的遮挡一致性训练，提高冻结 DINOv2 特征的鲁棒性？

### Question 3

一致性损失的权重 λ 应该如何选择？

---

# 2. 方法概览

整体结构如下：

```text
                 Clean Image
                     │
                     ↓
              Frozen DINOv2
                  ViT-S/14
                     │
                     ↓
             Trainable Adapter
                 384 → 256
                     │
            ┌────────┴────────┐
            ↓                 ↓
       Classifier       Consistency Loss
            ↑                 ↑
            │                 │
             ─────────────────
                     ↑
                     │
               Occluded Image
```

训练过程中：

* DINOv2 Backbone 完全冻结
* 只训练 Adapter 与分类器
* 干净图用于正常分类
* 干净图与遮挡图共同参与一致性约束

---

# 3. DINOv2 Backbone

项目使用：

```text
DINOv2 ViT-S/14
```

作为主要视觉特征提取器。

DINOv2 输出：

```text
384-dimensional feature
```

实验中不对 Backbone 进行 Fine-tuning，而是：

```text
Freeze DINOv2
```

这样能够：

* 保留预训练视觉特征
* 降低训练成本
* 减少可训练参数
* 更直接地研究冻结视觉表示的稳定性

---

# 4. Lightweight Adapter

在 DINOv2 输出特征后加入轻量 Adapter：

```text
Linear(384, 256)
       ↓
LayerNorm
       ↓
GELU
       ↓
Dropout(0.1)
       ↓
Linear(256, 10)
```

其中：

```text
384 → 256
```

负责对 DINOv2 特征进行任务适配。

DINOv2 保持冻结，只更新：

```text
Adapter
+
Classifier
```

总可训练参数约：

```text
101.6K
```

因此相较于全量 Fine-tuning，训练成本明显更低。

---

# 5. Occlusion Consistency Regularization

本项目的核心改进是：

**Occlusion Consistency Regularization**

对于原始图像：

```text
x
```

随机生成遮挡版本：

```text
x_occ
```

两者经过同一个冻结 DINOv2 和 Adapter：

```text
x
 ↓
DINOv2
 ↓
Adapter
 ↓
h
```

```text
x_occ
 ↓
DINOv2
 ↓
Adapter
 ↓
h_occ
```

训练目标要求：

> 同一张图像在遮挡前后的高层表示不要发生过大的变化。

---

## Loss Function

分类损失：

```text
L_cls
```

用于正常类别学习。

一致性损失：

```text
L_consistency
```

使用干净图与遮挡图 Adapter 表征之间的余弦距离。

最终损失：

```text
L_total = L_cls + λ · L_consistency
```

其中：

```text
λ
```

控制鲁棒性约束强度。

主实验使用：

```text
λ = 0.5
```

并通过消融实验进一步比较：

```text
λ = 0
λ = 0.1
λ = 0.5
λ = 1.0
```

---

# 6. 为什么一致性 Loss 加在 Adapter 上

一个重要设计点是：

如果直接在完全冻结的 DINOv2 输出上计算一致性 Loss，而后面没有可训练映射：

```text
Consistency Loss
↓
Frozen Backbone
↓
参数无法更新
```

Loss 无法改变 Backbone 的表示。

因此，本项目将一致性约束作用在：

```text
Trainable Adapter Output
```

上。

这样梯度可以更新 Adapter，使其学习：

```text
Clean Feature
≈
Occluded Feature
```

同时分类损失继续保证不同类别之间具有区分能力。

---

# 7. 数据集

实验使用：

```text
CIFAR-10
```

数据划分：

```text
Training Set    45,000
Validation Set   5,000
Test Set        10,000
```

由于 CIFAR-10 原图为：

```text
32 × 32
```

为适配预训练视觉模型，统一缩放到：

```text
224 × 224
```

训练阶段使用随机水平翻转。

验证与测试阶段不使用随机增强。

---

# 8. Baseline 模型

项目比较四种主要配置：

```text
ResNet50
ConvNeXt V2 Tiny
DINOv2 ViT-S/14
DINOv2 + Occlusion-Consistency Adapter
```

所有 Baseline 都采用统一下游训练协议。

需要说明的是：

> 不同 Backbone 的原始预训练数据和目标并不相同，因此实验比较的是这些公开预训练模型在统一下游任务中的综合表现，而不能把所有性能差异简单归因于 CNN 与 Transformer 架构本身。

---

# 9. 鲁棒性扰动设计

项目设计了四类常见视觉扰动。

每种扰动包含五个强度等级。

---

## Occlusion

随机遮挡图像的一部分。

| Level | Occlusion Ratio |
| ----- | --------------: |
| L1    |            0.10 |
| L2    |            0.20 |
| L3    |            0.30 |
| L4    |            0.40 |
| L5    |            0.50 |

训练一致性模型时：

```text
Occlusion Ratio ∈ [0.10, 0.40]
```

随机采样。

---

## Gaussian Noise

| Level |    σ |
| ----- | ---: |
| L1    | 0.05 |
| L2    | 0.10 |
| L3    | 0.15 |
| L4    | 0.20 |
| L5    | 0.25 |

---

## Gaussian Blur

| Level |    σ |
| ----- | ---: |
| L1    | 0.50 |
| L2    | 1.00 |
| L3    | 1.50 |
| L4    | 2.00 |
| L5    | 2.50 |

---

## Brightness Reduction

| Level | Brightness Factor |
| ----- | ----------------: |
| L1    |              0.80 |
| L2    |              0.60 |
| L3    |              0.40 |
| L4    |              0.25 |
| L5    |              0.10 |

---

# 10. 实验设置

主要训练配置：

```text
Epochs        20
Optimizer     AdamW
Learning Rate 3e-4
Weight Decay  1e-4
Batch Size    64
Random Seed   42
```

根据验证集：

```text
Clean Accuracy
```

保存最佳模型。

实验环境：

```text
GPU      NVIDIA RTX 4090
PyTorch  2.5.1 + CUDA 12.1
timm     1.0.28
```

鲁棒性测试使用固定扰动随机种子：

```text
1234
```

---

# 11. 遮挡实验结果

由于本方法直接针对遮挡进行训练，因此 Occlusion 是最主要的实验。

原始 DINOv2：

```text
Mean Occlusion Accuracy
82.66%
```

加入一致性 Adapter：

```text
Mean Occlusion Accuracy
85.05%
```

提升：

```text
+2.40 percentage points
```

不同遮挡强度下，相比原始 DINOv2 的提升分别为：

```text
L1   +0.45
L2   +1.12
L3   +2.32
L4   +3.68
L5   +4.41
```

可以观察到：

> 遮挡越严重，一致性训练带来的改善越明显。

这与方法最初的设计目标一致：

```text
减少视觉输入变化
        ↓
引起的特征漂移
```

---

# 12. 一个重要的失败案例

实验并不是所有情况下都优于其他模型。

在：

```text
50% Occlusion
```

条件下：

```text
ConvNeXt V2 Tiny
72.47%
```

而：

```text
DINOv2 + Ours
71.47%
```

仍然低：

```text
1.00 percentage point
```

这个结果说明：

> 只在 DINOv2 最终特征后进行一致性对齐，并不能完全解决 Transformer 内部的 Patch / Attention 鲁棒性问题。

Adapter 可以减少最终特征漂移，但无法恢复已经丢失的视觉信息，也没有直接修改 Transformer 内部注意力机制。

---

# 13. 其他扰动结果

虽然训练阶段只使用遮挡，但模型在其他扰动条件下也出现了一定变化。

---

## Gaussian Noise

原始 DINOv2：

```text
70.91%
```

DINOv2 + Ours：

```text
72.58%
```

提升：

```text
+1.67 percentage points
```

最强噪声条件下：

```text
41.75% → 44.80%
```

这表明 Adapter 学到的可能不仅是对：

```text
black occlusion pattern
```

的适应，也可能增强了部分特征稳定性。

---

## Gaussian Blur

DINOv2：

```text
97.56%
```

DINOv2 + Ours：

```text
97.75%
```

仅提升：

```text
+0.19
```

该任务存在明显天花板效应。

由于 CIFAR-10 原图仅为 32×32，放大后再进行模糊可能没有进一步破坏太多有效信息，因此该结果不能直接推广到高分辨率真实图像。

---

## Brightness

平均准确率：

```text
93.83% → 94.78%
```

提升：

```text
+0.95
```

不过在最暗条件下，其他模型仍可能具有优势。

这进一步说明：

> 视觉鲁棒性并不是一个单一指标，不同模型可能在不同扰动类型下表现出完全不同的强弱关系。

---

# 14. λ 消融实验

为了判断提升究竟来自：

```text
增加 Adapter 参数
```

还是：

```text
Consistency Learning
```

项目进行了 λ 消融实验。

| 配置          |     Clean | Occlusion Avg. | Corruption Avg. |
| ----------- | --------: | -------------: | --------------: |
| DINOv2      |     97.45 |          82.66 |           86.24 |
| λ = 0       |     97.53 |          83.19 |           86.48 |
| λ = 0.1     |     97.53 |          84.68 |           87.44 |
| **λ = 0.5** | **97.67** |      **85.05** |       **87.54** |
| λ = 1.0     |     97.57 |          84.92 |           86.97 |

其中：

```text
λ = 0
```

表示：

> 保留 Adapter，但完全不使用一致性损失。

结果显示：

```text
86.24% → 86.48%
```

只提升：

```text
0.24 percentage points
```

而加入中等强度一致性约束后：

```text
λ = 0.1 → 87.44%
λ = 0.5 → 87.54%
```

因此实验结果支持：

> 主要提升来自一致性训练，而不是单纯增加网络参数。

---

## λ 并不是越大越好

当：

```text
λ = 1.0
```

时，平均扰动准确率下降到：

```text
86.97%
```

一种可能解释是：

分类损失希望：

```text
不同类别分开
```

而一致性损失希望：

```text
同一图像的不同版本靠近
```

如果 λ 太大，过强的一致性目标可能损害类别区分能力。

需要注意：

```text
λ = 0.1
```

与：

```text
λ = 0.5
```

之间仅相差：

```text
0.10 percentage points
```

且当前每个配置只运行一次，因此不应把这一微小差异解释成稳定显著优势。

更合理的结论是：

> λ = 0.1 ~ 0.5 的中等一致性约束在当前实验中表现较好。

---

# 15. 一个重要观察：Clean Accuracy ≠ Robustness

实验中可以明显看到：

```text
Clean Accuracy
```

与：

```text
Robustness
```

并不是完全一致的。

例如：

ConvNeXt V2 Tiny 的干净准确率：

```text
94.47%
```

低于 DINOv2：

```text
97.45%
```

但在平均遮挡条件下：

```text
ConvNeXt V2   83.70%
DINOv2        82.66%
```

ConvNeXt V2 反而更高。

因此：

> 只报告标准测试集准确率可能掩盖模型的重要弱点。

这也是本项目同时报告：

```text
Clean Performance
+
Corruption Performance
```

的原因。

---

# 16. 项目结构

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

# 17. 核心模块

## `src/models.py`

实现：

* ResNet50 Backbone
* ConvNeXt V2 Tiny Backbone
* DINOv2 ViT-S/14
* Frozen Feature Extractor
* Lightweight Adapter
* Classifier

---

## `src/corruptions.py`

实现：

```text
Occlusion
Gaussian Noise
Gaussian Blur
Brightness Reduction
```

用于训练增强与统一鲁棒性测试。

---

## `src/data.py`

负责：

* CIFAR-10 数据加载
* Train / Validation Split
* Resize
* Data Augmentation
* DataLoader

---

## `src/engine.py`

实现：

* Training Loop
* Validation
* Classification Loss
* Consistency Loss
* Metric Aggregation

---

## `train_baseline.py`

训练：

```text
ResNet50
ConvNeXt V2 Tiny
DINOv2
```

Baseline。

---

## `train_consistency.py`

训练：

```text
Frozen DINOv2
+
Adapter
+
Occlusion Consistency Loss
```

---

## `evaluate_robustness.py`

统一评估：

```text
Clean
Occlusion
Noise
Blur
Brightness
```

条件下的 Top-1 Accuracy。

---

## `plot_results.py`

负责生成：

* Clean vs Corrupted Accuracy
* Occlusion Severity Curve
* λ Ablation Curve
* Model Comparison

等结果图。

---

# 18. 环境安装

推荐：

```text
Python 3.10+
```

克隆仓库：

```bash
git clone https://github.com/zhu-yaqi/dinov2-robustness.git
cd dinov2-robustness
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

# 19. 环境检查

```bash
python check_env.py
```

用于检查：

* Python
* PyTorch
* CUDA
* GPU
* 相关依赖

---

# 20. 数据准备

```bash
python download_data.py
```

数据集不会直接提交到 GitHub。

---

# 21. 训练 Baseline

```bash
python train_baseline.py
```

---

# 22. 训练一致性模型

```bash
python train_consistency.py
```

---

# 23. 鲁棒性评估

```bash
python evaluate_robustness.py
```

用于在：

```text
Clean
Occlusion
Gaussian Noise
Gaussian Blur
Brightness
```

条件下统一评估模型。

---

# 24. 结果可视化

```bash
python plot_results.py
```

---

# 25. 完整实验流程

```bash
python run_all.py
```

具体实验设计可参考：

```text
configs/experiment_plan.md
```

---

# 26. 对实验结果的理解

这个项目最重要的结论并不是：

> DINOv2 + Adapter 在所有条件下都优于其他模型。

更准确的结论是：

> 在冻结 DINOv2 主干网络、只训练少量参数的情况下，一致性 Adapter 改善了 DINOv2 较明显的遮挡短板，同时对噪声等其他扰动表现出一定迁移效果。

同时：

* ConvNeXt V2 在极端遮挡下仍可能更强
* ResNet50 在部分特殊扰动条件下仍有优势
* DINOv2 在模糊条件下已经存在明显天花板
* 更高的标准分类准确率不意味着更高的鲁棒性

因此视觉鲁棒性应该被视为：

```text
a set of failure modes
```

而不是一个单一的平均分数。

---

# 27. 与认知科学的联系

本项目受到人类视觉中：

```text
Object Constancy
```

概念的启发。

人类在视觉输入发生一定变化时，通常仍能维持较稳定的对象判断。

本方法将这一现象简化为一个可训练约束：

> 同一对象在遮挡前后的高层表示不应该发生过大变化。

但需要强调：

本项目并不声称模型因此获得了与人类相同的视觉机制。

当前方法：

* 不理解遮挡物是什么
* 不补全被遮挡区域
* 不模拟大脑反馈连接
* 不包含人类被试或脑成像数据

因此更准确的说法是：

> 方法受到认知科学中的稳定表征思想启发，而不是对人类视觉机制的直接模拟。

---

# 28. 局限性

当前实验仍存在以下限制：

* 只使用 CIFAR-10
* CIFAR-10 原始分辨率仅为 32×32
* 每个配置只运行一个 Random Seed
* 尚未报告多次重复实验的均值与标准差
* 使用黑色方块模拟遮挡，与自然遮挡存在差异
* 不同 Backbone 的预训练数据规模和训练目标并不统一
* 只对齐最终特征，没有直接处理 Transformer 中间层或 Patch Token
* 没有人类行为数据进行直接比较

因此目前实验结果不应被解释为：

> 本方法全面解决了视觉鲁棒性问题。

---

# 29. 后续工作

未来可以进一步研究：

* CIFAR-10-C
* ImageNet-C
* Natural Occlusion Dataset
* 多随机种子重复实验
* Mean ± Standard Deviation
* 对齐 DINOv2 中间层特征
* Patch Token Consistency
* Attention Robustness
* 更真实的自然遮挡
* 遮挡分支额外加入 Classification Loss
* 人类与模型错误模式比较

---

# 30. 项目收获

通过这个项目，我对视觉鲁棒性实验有了几个更具体的认识：

### 1. 准确率高不代表鲁棒性强

不同模型在 Clean 和 Corrupted 条件下的排序可能明显变化。

### 2. 消融实验非常重要

如果只比较：

```text
DINOv2
vs.
DINOv2 + Adapter + Consistency
```

无法判断提升究竟来自：

```text
Adapter
```

还是：

```text
Consistency Loss
```

加入：

```text
λ = 0
```

这一对照后，才能更清楚地分析改进来源。

### 3. 失败结果也有价值

极端遮挡下 ConvNeXt V2 仍然优于本方法，反而帮助定位当前方案的局限：

```text
最终层特征对齐
```

不能完全解决：

```text
Transformer 内部 Patch / Attention
```

相关问题。

### 4. 研究不只是追求最高数字

相比只报告最好的 Accuracy，更重要的是：

```text
提出问题
↓
设计方法
↓
设置对照
↓
完成实验
↓
分析什么时候有效
↓
分析什么时候失效
```

---

# 技术栈

```text
Python
PyTorch
Torchvision
timm
DINOv2
Vision Transformer
ResNet50
ConvNeXt V2
Matplotlib
```

涉及方向：

```text
Computer Vision
Self-Supervised Learning
Vision Transformer
Robustness Evaluation
Image Corruption
Consistency Regularization
Transfer Learning
Parameter-Efficient Adaptation
Ablation Study
```
