# 实验计划与记录模板

## 固定设置

- Dataset: CIFAR-10
- Input size: 224 × 224
- Train/Val split seed: 42
- Optimizer: AdamW
- Epochs: 20
- Learning rate: 3e-4
- Weight decay: 1e-4
- Backbone: frozen
- Classes: 10

## 主实验模型

| ID | Model | Trainable Part | Role |
|---|---|---|---|
| B1 | ResNet50 | Linear head | classical CNN baseline |
| B2 | ConvNeXt V2 Tiny | Linear head | recent CNN baseline |
| B3 | DINOv2 ViT-S/14 | Linear head | SOTA-style main backbone |
| Ours | DINOv2 ViT-S/14 | Adapter + head | occlusion consistency |

## Ours 设置

- Adapter dimension: 256
- lambda_consistency: 0.5
- Training occlusion area ratio: Uniform(0.10, 0.40)

## 消融

- lambda = 0.0
- lambda = 0.1
- lambda = 0.5
- lambda = 1.0

## Robustness Tests

- clean
- occlusion: 10%, 20%, 30%, 40%, 50%
- Gaussian noise sigma: 0.05, 0.10, 0.15, 0.20, 0.25
- Gaussian blur sigma: 0.5, 1.0, 1.5, 2.0, 2.5
- brightness factor: 0.8, 0.6, 0.4, 0.25, 0.10

## 每次正式实验记录

- Date:
- Machine:
- CPU:
- GPU:
- GPU memory:
- OS:
- Python:
- torch:
- torchvision:
- timm:
- DINOv2 source:
- batch size:
- num workers:
- seed:
- epoch:
- lr:
- training time:
- best val accuracy:
- clean test accuracy:
- mean corruption accuracy:
- notes:

## 写 Discussion 时逐题回答

1. DINOv2 是否比 ResNet50 和 ConvNeXt V2 更抗遮挡？
2. Consistency 是否提高遮挡鲁棒性？
3. 是否牺牲 clean accuracy？
4. lambda=0 的 Adapter 本身带来多少变化？
5. 哪个 lambda 的 trade-off 最合理？
6. 只用遮挡训练，能否迁移到 noise / blur / brightness？
7. 哪个结果和原假设不一致？
8. CIFAR-10 从 32×32 放大到 224×224 带来什么局限？
9. 冻结 backbone 与全量 fine-tuning 相比有什么优缺点？
10. 为什么这些结果还不能证明模型“像人脑”？
