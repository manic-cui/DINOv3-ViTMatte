import torch
import wget
import os
from detectron2.config import LazyConfig, instantiate

# --- 辅助函数，用于预处理 DINOv1 权重 ---
def preprocess_dinov1_weights(original_weights):
    new_model = {}
    for k, v in original_weights.items():
        # DINOv1 权重里没有 'backbone.' 前缀，我们需要加上
        new_key = 'backbone.' + k
        if 'patch_embed.proj.weight' in k:
            embed_dim, _, patch_size, _ = v.shape
            new_weight = torch.zeros(embed_dim, 4, patch_size, patch_size, dtype=v.dtype)
            new_weight[:, :3, :, :] = v
            new_model[new_key] = new_weight
        else:
            new_model[new_key] = v
    return new_model

def main():
    # --- 1. 创建一个标准的虚拟输入 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(1, 4, 512, 512).to(device)
    print("="*80)
    print(f"创建了一个虚拟输入张量，维度: {dummy_input.shape}，设备: {device}")
    print("="*80 + "\n")

    # --- 2. 加载和测试原始的 DINOv1 主干网络 ---
    print("--- 正在加载和测试原始 DINOv1 (ViT-S) 主干网络 ---")
    try:
        # 加载基础模型配置
        cfg = LazyConfig.load("configs/common/model.py")
        # **重要修复**: 明确指定输入通道为4
        cfg.model.backbone.in_chans = 4
        
        original_backbone = instantiate(cfg.model.backbone).to(device).eval()

        # 下载并预处理 DINOv1 权重
        dino1_url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
        dino1_filename = 'dino_deitsmall16_pretrain.pth'
        if not os.path.exists(dino1_filename):
            print(f"正在下载原始 DINOv1 权重...")
            wget.download(dino1_url)
        
        original_weights = torch.load(dino1_filename, map_location='cpu')
        processed_weights = preprocess_dinov1_weights(original_weights)
        
        original_backbone.load_state_dict(processed_weights, strict=False)
        print("\n原始 DINOv1 主干网络加载成功！")
        
        with torch.no_grad():
            output_dino1 = original_backbone(dummy_input)
        
        print("\n--- 原始 DINOv1 输出分析 ---")
        print(f"输出特征图维度: {output_dino1.shape}")
        print(f"输出特征图均值 (Mean): {output_dino1.mean().item():.6f}")
        print(f"输出特征图标准差 (Std): {output_dino1.std().item():.6f}")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n加载原始 DINOv1 模型时出错: {e}")
        print("="*80 + "\n")


    # --- 3. 加载和测试新的 DINOv3 主干网络 ---
    print("--- 正在加载和测试新的 DINOv3 (ViT-S) 主干网络 ---")
    try:
        # 加载为 DINOv3 修改的配置
        cfg_dino3 = LazyConfig.load("configs/ViTMatte_S_100ep.py") # 确保您有这个文件
        
        # 实例化修改后的 ViT backbone (带 RoPE)
        dinov3_backbone = instantiate(cfg_dino3.model.backbone)
        dinov3_backbone.cuda()
        dinov3_backbone.eval()

        # 加载您之前预处理好的 DINOv3 权重
        dinov3_weights_path = './dinov3_vit_s_fna.pth'
        if not os.path.exists(dinov3_weights_path):
             raise FileNotFoundError(f"找不到 DINOv3 权重文件: {dinov3_weights_path}。请先运行 preprocess_dinov3.py。")

        dinov3_weights = torch.load(dinov3_weights_path, map_location='cpu')
        
        # 加载权重 (strict=False)
        dinov3_backbone.load_state_dict(dinov3_weights, strict=False)

        print("\n新的 DINOv3 主干网络加载成功！")

        # 前向传播并打印结果
        with torch.no_grad():
            output_dino3 = dinov3_backbone(dummy_input)

        print("\n--- 新的 DINOv3 输出分析 ---")
        print(f"输出特征图维度: {output_dino3.shape}")
        print(f"输出特征嵌入维度 (通道数): {output_dino3.shape[1]}")
        print(f"输出特征图均值 (Mean): {output_dino3.mean().item():.6f}")
        print(f"输出特征图标准差 (Std): {output_dino3.std().item():.6f}")
        print("="*80)

    except Exception as e:
        print(f"\n加载 DINOv3 模型时出错: {e}")
        print("="*80)

if __name__ == '__main__':
    main()