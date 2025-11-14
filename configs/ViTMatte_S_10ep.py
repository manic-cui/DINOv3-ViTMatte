from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader
from detectron2.config import LazyCall as L
from modeling import ViT
model.backbone = L(ViT)(
    embed_dim=384,
    depth=12,
    num_heads=6,
    #通道数4
    in_chans=4,
    use_rel_pos=True,  # <-- 关键：启用RoPE (旋转位置编码)
    window_size=14,
    window_block_indexes=[
        # 2, 5, 8, 11 for ViT-B
        # 1, 4, 7, 10 for ViT-L
        1, 4, 7, 10,  # 这里的配置可以根据你的实验需求调整
    ],
    residual_block_indexes=[
        # 2, 5, 8, 11 for ViT-B
        # 1, 4, 7, 10 for ViT-L
        1, 4, 7, 10, # 这里的配置可以根据你的实验需求调整
    ],
)

# 1. 训练总时长：保持10个epoch
train.max_iter = int(43100 / 16 / 2 * 10) 
# 每5个epoch保存一次模型
train.checkpointer.period = int(43100 / 16 / 2 * 2)
optimizer.lr = 5e-4
lr_multiplier.scheduler.values=[1.0, 0.5, 0.2]
lr_multiplier.scheduler.milestones=[int(43100 / 16 / 2 * 8), int(43100 / 16 / 2 * 9.5)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

# 5. 使用新的输出目录以作区分
train.init_checkpoint = '/data/cuimanni/vitmatte_result/output_of_train/dinov3_vit_s_fna.pth'
train.output_dir = '/data/cuimanni/vitmatte_result/output_of_train/ViTMatte_S_10ep_new_1'

# 保持不变
dataloader.train.batch_size=16
dataloader.train.num_workers=2