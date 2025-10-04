# vit.py

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, Mlp, trunc_normal_

from .backbone import Backbone
from .utils import PatchEmbed, get_abs_pos, window_partition, window_unpartition
from .rope_position_encoding import RopePositionEmbedding

logger = logging.getLogger(__name__)

__all__ = ["ViT"]


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    # Adjust dtype for RoPE application if needed
    x_dtype = x.dtype
    rope_dtype = sin.dtype
    x = x.to(dtype=rope_dtype)
    
    # Apply RoPE
    out = (x * cos) + (rope_rotate_half(x) * sin)
    return out.to(dtype=x_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    """
    DINOv3 SelfAttention block adapted to ViTMatte's API, with RoPE integrated.
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
                embed_dim=self.head_dim, # DINOv3 RoPE is per head
                num_heads=self.num_heads,
                base=100.0,
                normalize_coords="separate",
                dtype=torch.float32, # Use float32 for stability
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
    """Transformer blocks with LayerScale and DINOv3-style RoPE attention"""

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
        # DINOv3-specific
        layerscale_init=1e-5,
        **kwargs, # absorb other unused args
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
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.ls1 = LayerScale(dim, init_values=layerscale_init) if layerscale_init else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.ls2 = LayerScale(dim, init_values=layerscale_init) if layerscale_init else nn.Identity()

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)
        
        # Window logic from ViTMatte
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x_norm, pad_hw = window_partition(x_norm, self.window_size)

        attn_out = self.attn(x_norm)
        
        if self.window_size > 0:
            attn_out = window_unpartition(attn_out, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(self.ls1(attn_out))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x


class ViT(Backbone):
    """
    This module implements a Vision Transformer backbone compatible with DINOv3 weights.
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
        use_abs_pos=True,
        use_rel_pos=False,
        window_size=0,
        window_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        # DINOv3-specific
        layerscale_init=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
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
            )
            if use_act_checkpoint:
                # you may need to import checkpoint_wrapper from fairscale
                from fairscale.nn.checkpoint import checkpoint_wrapper
                block = checkpoint_wrapper(block)
            self.blocks.append(block)
        
        # Add the final norm layer
        self.norm = norm_layer(embed_dim)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

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
        # ViTMatte needs a custom loader to handle weights that don't match
        # This is especially true when loading DINOv3 weights.
        super().load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        for blk in self.blocks:
            x = blk(x)

        # Apply the final norm layer before output
        x = self.norm(x)
        
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs['last_feat']