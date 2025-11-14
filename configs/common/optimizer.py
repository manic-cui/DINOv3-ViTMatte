from detectron2 import model_zoo
from functools import partial

def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12, head_lr_mult=1.0):
    """
    为不同的ViT块和解码器（分类头）计算学习率衰减率。
    Args:
        name (string): 参数名称。
        lr_decay_rate (float): ViT主干网络的基础学习率衰减率。
        num_layers (int): ViT块的数量。
        head_lr_mult (float): 解码器（分类头）的学习率乘子。

    Returns:
        给定参数的学习率乘子。
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        return lr_decay_rate ** (num_layers + 1 - layer_id)
    if name.startswith("decoder"):
        return head_lr_mult
    
    # 对于其他参数（如果有的话），返回默认乘子1.0
    return 1.0

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW


optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.65, head_lr_mult=3)

optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}