from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader

# 1. 训练总时长：保持10个epoch
train.max_iter = int(43100 / 16 / 2 * 10) 
# 每2个epoch保存一次模型
train.checkpointer.period = int(43100 / 16 / 2 * 2)
optimizer.lr = 5e-4
lr_multiplier.scheduler.values=[1.0, 0.5, 0.2]
lr_multiplier.scheduler.milestones=[int(43100 / 16 / 2 * 8), int(43100 / 16 / 2 * 9.5)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

# 5. 使用新的输出目录以作区分
train.init_checkpoint = '/home/mannicui/ViTMatte/dinov3_vit_s_fna.pth'
train.output_dir = './output_of_train/ViTMatte_S_10ep_9'

# 保持不变
dataloader.train.batch_size=16
dataloader.train.num_workers=2