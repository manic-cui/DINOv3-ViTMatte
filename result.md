# DINOv3-ViTMatte 改进总结报告 (最终修正版)

## 概述

本项目旨在将原始的 ViTMatte 模型与先进的 DINOv3 架构相结合，以提升图像抠图性能。核心改进在于用 DINOv3 的 Vision Transformer (ViT) 替换了 ViTMatte 中原有的 ViT 主干网络，并引入了旋转位置编码 (RoPE) 等 DINOv3 的关键特性。

---

## 主要文件修改

### 1. **预训练模型处理 (`preprocess.py`)**

为了让 DINOv3 的预训练权重能够适配 ViTMatte，我修改了 `preprocess.py` 脚本。关键改动包括：
* **权重键名适配**：脚本会自动为 DINOv3 的权重键添加 `backbone.` 前缀，以匹配 ViTMatte 的模型结构。
* **输入通道适配**：`patch_embed` 层的权重被从3通道（RGB）扩展到4通道，以适应 ViTMatte 的输入（RGB + Trimap）。
* **移除不用的权重**：脚本会跳过 ViTMatte 模型中不存在的参数，如 `cls_token` 和 `mask_token`。
* **保存格式**：处理后的权重被保存在一个 `{"model": ...}` 的字典中，这是 Detectron2 框架推荐的格式。

### 2. **主干网络 (`modeling/backbone/vit.py`)**

这是最重要的修改。我用一个与 DINOv3 权重兼容的新 `ViT` 类替换了原有的实现。关键改动包括：

* **注意力机制 (`Attention` 类)**:
    * 引入了旋转位置编码 (`RopePositionEmbedding`)，在注意力计算前对查询 (`q`) 和键 (`k`) 进行编码。这是 DINOv3 的核心组件之一。
    * 使用了 `F.scaled_dot_product_attention` 来优化注意力计算。
* **Transformer 块 (`Block` 类)**:
    * 引入了 `LayerScale`，这是 DINOv3 中用于稳定训练和提升性能的技术。
* **ViT 主类 (`ViT` 类)**:
    * 在模型末尾增加了一个最终的归一化层 (`self.norm`)，以匹配 DINOv3 的架构。
    * 重写了 `load_state_dict` 方法，允许以非严格模式 (`strict=False`) 加载权重，这对于加载通道数不匹配的预训练模型至关重要。

### 3. **旋转位置编码 (`modeling/backbone/rope_position_encoding.py`)**

这是一个新增文件，实现了 `RopePositionEmbedding` 模块，用于计算二维图像的旋转位置编码，是 DINOv3 架构的关键部分。

### 4. **配置文件 (`configs/ViTMatte_S_100ep.py`)**

我更新了训练配置文件，以反映新的模型和训练策略。

* **优化器策略**: 引入了 **ViT 的分层学习率衰减** (`get_vit_lr_decay_rate`)。越靠近底层的网络（如 patch embedding），其学习率越低，而越靠近顶层的网络，学习率越高。衰减率为 `0.65`。
* **训练周期**: 将训练周期设定为 **10个 epoch**。


---

## 训练参数总结
训练时使用的主要参数如下：



* **优化器与学习率**:
    * `optimizer`: **AdamW**
    * `optimizer.lr`: 5e-4
    * **分层学习率衰减 (Layer-wise LR Decay)**:
        * `lr_decay_rate`: **0.65**
        * `num_layers`: 12
        * 此策略为 Transformer 的不同层设置了不同的学习率。
        * dinov3的学习率和分类头的学习率不同，分类头的学习率是dinov3的五倍
    * **学习率调度**:
        * `lr_multiplier.scheduler.values`: `[1.0, 0.5, 0.2]`
        * `lr_multiplier.scheduler.milestones`: 在第 **8** 和 **9.5** 个 epoch 时进行学习率衰减。
        * `lr_multiplier.warmup_length`: 1000 次迭代。

* **训练过程**:
    * `train.max_iter`: 13470 (对应 10 epoch)
    * `train.checkpointer.period`: 2694 (每 2 个 epoch 保存一次模型)
    * `dataloader.train.batch_size`: 16
    * `dataloader.train.num_workers`: 2


---

## 结果

* Unknown Region: MSE: 0.006098047653826718 SAD: 29.279445091796877 

