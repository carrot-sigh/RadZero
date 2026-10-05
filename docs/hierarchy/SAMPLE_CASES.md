# Hierarchy Sample Cases

本文档使用 canonical hierarchy test split 中的三个真实样本，说明数据标注
以及 `Stage 1 -> Stage 2 -> Stage 3` 的输入输出关系。

原始图像来自 MIMIC-CXR-JPG，Anatomy bbox 来自 Chest ImaGenome，Global
标签来自 CheXpert。文中的 Anatomy 框是 GT 标注，用于解释数据结构；正式
推理时由 Stage 1 预测 ROI，不读取 GT bbox。

样本可视化文件由 manifest 生成，位于：

```text
/home/user/gxq/hierarchy-samples/
```

## Sample 1: 双侧气胸与右侧积液

可视化：`50295453.jpg`

```text
study_id: 50295453
split: test
Anatomy boxes: 36
```

Region-Finding 正例：

```text
right lung:
  lung opacity
  pleural effusion
  pneumothorax

right apical zone:
  pneumothorax

right costophrenic angle:
  lung opacity
  pleural effusion

left lung:
  pneumothorax

left apical zone:
  pneumothorax
```

Global 标签：

```text
Pneumothorax = 1
Support Devices = 1
No Finding = 0
Pleural Effusion = 0
```

区域 Finding 和 Global CheXpert 标签不是一一映射关系。区域上存在
`pleural effusion`，并不必然意味着 Global `Pleural Effusion=1`。

## Sample 2: 肺部病灶、肺部阴影与心影增大

可视化：`50683678.jpg`

Region-Finding 正例：

```text
right lung:
  lung lesion
  lung opacity
  mass/nodule (not otherwise specified)
  atelectasis
  pleural effusion

right apical zone:
  lung lesion
  lung opacity
  mass/nodule (not otherwise specified)

right lower lung zone:
  atelectasis
  lung opacity
  pleural effusion

right costophrenic angle:
  atelectasis
  lung opacity
  pleural effusion

left lung:
  lung lesion
  lung opacity
  mass/nodule (not otherwise specified)

left apical zone:
  lung lesion
  lung opacity
  mass/nodule (not otherwise specified)

cardiac silhouette:
  enlarged cardiac silhouette
```

明确 negative：

```text
left lung: pleural effusion = 0, pneumothorax = 0
left costophrenic angle: pleural effusion = 0
```

Global 标签：

```text
Atelectasis = 1
Cardiomegaly = 1
Lung Lesion = 1
Pleural Effusion = 1
Pneumothorax = 0
Support Devices = 1
No Finding = 0
```

## Sample 3: 肺部阴影、血管充血与心影增大

可视化：`53955641.jpg`

Region-Finding 正例：

```text
right lung:
  lung opacity
  vascular congestion

right hilar structures:
  lung opacity
  vascular congestion

left lung:
  lung opacity
  vascular congestion

left hilar structures:
  lung opacity
  vascular congestion

cardiac silhouette:
  enlarged cardiac silhouette
```

明确 negative：

```text
right lung: pleural effusion = 0, consolidation = 0, pneumothorax = 0
left lung: pleural effusion = 0, consolidation = 0, pneumothorax = 0
right/left costophrenic angle: pleural effusion = 0
```

Global 标签：

```text
Cardiomegaly = 1
Consolidation = 0
Pleural Effusion = 0
No Finding = 0
```

## 三阶段数据流

### Stage 1: Anatomy ROI

输入：

```text
CXR image
```

训练监督：

```text
GT Anatomy bbox -> 37x37 patch mask
```

模型输出：

```text
predicted_anatomy_masks [B, 36, 1369]
```

部署时使用预测 Anatomy mask，不使用 GT bbox。

### Stage 2: Region Finding

输入：

```text
full-image visual features
Stage 1 predicted Anatomy masks
41 Finding text queries
```

以 Sample 3 为例，模型会计算：

```text
right lung x lung opacity
right lung x vascular congestion
left lung x lung opacity
left lung x vascular congestion
cardiac silhouette x enlarged cardiac silhouette
```

输出：

```text
finding_logits [B, 36, 41]
finding_attention
finding_evidence
valid_pair_mask
```

训练监督：

```text
finding_targets [B, 36, 41]
finding_valid_mask [B, 36, 41]
```

只有明确的 Anatomy-Finding `yes/no` pair 参与 loss；缺失或冲突标注不当作
negative。

### Stage 3: Global

输入：

```text
Stage 2 finding evidence
Stage 2 finding logits
RadZero global visual evidence
14 Global text queries
```

输出：

```text
global_logits [B, 14]
```

训练监督：

```text
global_targets [B, 14]
global_valid_mask [B, 14]
```

例如 Sample 2 的 Global 监督包括 `Atelectasis=1`、`Cardiomegaly=1`、
`Lung Lesion=1`、`Pleural Effusion=1`、`Pneumothorax=0` 和
`Support Devices=1`。

## 完整部署链路

```text
CXR
  -> Stage 1 predicted Anatomy ROI
  -> Stage 2 regional Finding
  -> Stage 3 global disease prediction
```

训练时各阶段需要各自的监督标签；validation/test 的模型 forward 可以只
接收图像，标签只在模型输出之后用于计算指标。
