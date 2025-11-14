# vit.py (已加入卷积颈功能)

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, Mlp, trunc_normal_
import fvcore.nn.weight_init as weight_init

# 从 detectron2 导入必要的模块
from detectron2.layers import CNNBlockBase, Conv2d, get_norm

from .backbone import Backbone
from .utils import PatchEmbed, get_abs_pos, window_partition, window_unpartition
from .rope_position_encoding import RopePositionEmbedding

logger = logging.getLogger(__name__)

__all__ = ["ViT"]


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    x_dtype = x.dtype
    rope_dtype = sin.dtype
    x = x.to(dtype=rope_dtype)
    out = (x * cos) + (rope_rotate_half(x) * sin)
    return out.to(dtype=x_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ START: 从原始 ViTMatte 添加回来的卷积颈 (Convolution Neck) 模块 +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ResBottleneckBlock(CNNBlockBase):
    """
    标准的 bottleneck 残差模块，不包含最后的激活层。
    包含三个卷积层，核大小分别为 1x1, 3x3, 1x1。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            bottleneck_channels (int): "瓶颈" 3x3 卷积层的输出通道数。
            norm (str or callable): 所有卷积层的归一化方法。
            act_layer (callable): 所有卷积层的激活函数。
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # 最后一个 norm 层零初始化
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ END: 从原始 ViTMatte 添加回来的卷积颈 (Convolution Neck) 模块 +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Attention(nn.Module):
    """
    DINOv3 的自注意力模块，适配了 ViTMatte 的 API，并集成了 RoPE。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        input_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rope = RopePositionEmbedding(
                embed_dim=self.head_dim,
                num_heads=self.num_heads,
                base=100.0,
                normalize_coords="separate",
                dtype=torch.float32,
            )

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        N = H * W
        x_flat = x.reshape(B, N, C)

        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_rel_pos:
            sin, cos = self.rope(H=H, W=W)
            q = rope_apply(q, sin, cos)
            k = rope_apply(k, sin, cos)

        x = F.scaled_dot_product_attention(q, k, v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x.reshape(B, H, W, C)


class Block(nn.Module):
    """带有 LayerScale 和 DINOv3 风格 RoPE 注意力的 Transformer 模块"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        window_size=0,
        input_size=None,
        layerscale_init=1e-5,
        use_residual_block=False, # <--- 添加回来的参数
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.ls1 = LayerScale(dim, init_values=layerscale_init) if layerscale_init else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.ls2 = LayerScale(dim, init_values=layerscale_init) if layerscale_init else nn.Identity()

        self.window_size = window_size

        # +++ 添加回来的卷积颈实例化逻辑 +++
        self.use_residual_block = use_residual_block
        if use_residual_block:
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)
        
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x_norm, pad_hw = window_partition(x_norm, self.window_size)

        attn_out = self.attn(x_norm)
        
        if self.window_size > 0:
            attn_out = window_unpartition(attn_out, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(self.ls1(attn_out))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        # +++ 添加回来的卷积颈前向传播逻辑 +++
        if self.use_residual_block:
            # permute 从 (B, H, W, C) -> (B, C, H, W) 以适配卷积层
            # 然后再 permute 回来
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


class ViT(Backbone):
    """
    实现了与 DINOv3 权重兼容的 Vision Transformer 骨干网络。
    """
    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(), # <--- 添加回来的参数
        use_act_checkpoint=False,
        out_feature="last_feat",
        layerscale_init=1e-5,
        **kwargs,
    ):
        super().__init__()
        # self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # if use_abs_pos:
        #     num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        #     num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
        #     self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        # else:
        #     self.pos_embed = None
        self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                layerscale_init=layerscale_init,
                use_residual_block=i in residual_block_indexes, # <--- 将参数传递给 Block
            )
            if use_act_checkpoint:
                from fairscale.nn.checkpoint import checkpoint_wrapper
                block = checkpoint_wrapper(block)
            self.blocks.append(block)
        
        self.norm = norm_layer(embed_dim)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_state_dict(self, state_dict, strict=True):
        # 使用非严格模式加载，以忽略卷积颈的权重（因为预训练模型中没有）
        super().load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.patch_embed(x)


        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs['last_feat']