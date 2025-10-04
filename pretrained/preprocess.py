# preprocess.py

import torch
import wget
import os

def adapt_dinov3_for_vitmatte(dinov3_model_path, output_name, embed_dim, patch_size=16):
    """
    Downloads, adapts, and renames DINOv3 weights for ViTMatte.
    """
    # 1. Load the official DINOv3 model weights
    dinov3_state_dict = torch.load(dinov3_model_path, map_location="cpu")
    
    if 'student' in dinov3_state_dict:
        dinov3_state_dict = dinov3_state_dict['student']

    new_model_state_dict = {}

    # 2. Rename keys and adapt weights
    for k, v in dinov3_state_dict.items():
        # DINOv3 keys are like 'blocks.0.norm1.weight'
        # ViTMatte expects 'backbone.blocks.0.norm1.weight'
        new_key = 'backbone.' + k

        # Adapt patch embedding for 4-channel input
        if 'patch_embed.proj.weight' in k:
            print(f"Adapting patch embedding layer: {k} -> {new_key}")
            # DINOv3 is 3 channels, ViTMatte is 4
            new_weight = torch.zeros(v.shape[0], 4, v.shape[2], v.shape[3], dtype=v.dtype, device=v.device)
            new_weight[:, :3, :, :] = v
            new_model_state_dict[new_key] = new_weight
            print(" -> Adapted shape:", new_weight.shape)
            # Adapt bias if it exists
            if 'patch_embed.proj.bias' in dinov3_state_dict:
                 new_model_state_dict['backbone.patch_embed.proj.bias'] = dinov3_state_dict['patch_embed.proj.bias']
            continue
        
        # Skip tokens that ViTMatte doesn't use
        elif 'cls_token' in k or 'mask_token' in k or 'storage_tokens' in k:
            print(f"Skipping unused parameter: {k}")
            continue
        
        # The keys for LayerScale (ls1, ls2) and the final norm in DINOv3 already match
        # the names we've used in our revised vit.py (ls1, ls2, norm),
        # so no special renaming is needed for them. Just add the prefix.
        
        new_model_state_dict[new_key] = v

    # 3. Save the adapted model
    output_path = f"{output_name}.pth"
    torch.save({"model": new_model_state_dict}, output_path) # Save in a format detectron2 prefers
    print(f"\nSuccessfully preprocessed and saved adapted DINOv3 weights to: {output_path}")

if __name__ == "__main__":
    downloaded_filename = '/home/mannicui/ViTMatte/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth' # Make sure this path is correct
    
    if not os.path.exists(downloaded_filename):
        print(f"DINOv3 model not found at {downloaded_filename}. Please download it first.")
    else:
        # Parameters for ViT-Small (embed_dim=384)
        adapt_dinov3_for_vitmatte(
            dinov3_model_path=downloaded_filename,
            output_name='dinov3_vit_s_fna', # Use a new name
            embed_dim=384
        )