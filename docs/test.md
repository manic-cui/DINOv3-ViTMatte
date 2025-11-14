## Test

### Test Dataset

Get preprocessed Composition-1k Dataset from [MatteFormer](https://github.com/webtoon/matteformer).

### Inference

Run

```
CUDA_VISIBLE_DEVICES=2 python inference.py \
  --config-dir configs/ViTMatte_S_10ep.py \
  --checkpoint-dir /data/cuimanni/vitmatte_result/output_of_train/ViTMatte_S_10ep_new_1/model_final.pth \
  --inference-dir /data/cuimanni/vitmatte_result/ViTMatte_S_10ep_new_1 \
  --data-dir /data/cuimanni/Composition-1k-testset
```

to infer on Composition-1k.

### Evaluation

NOTE:  The final quantitative results of ViTMatte is NOT evaluted by `evaluation.py`. We use official matlab code from [DIM](https://github.com/foamliu/Deep-Image-Matting) for fair comparision.

Run

```
CUDA_VISIBLE_DEVICES=2 python evaluation.py \
    --pred-dir /data/cuimanni/vitmatte_result/ViTMatte_S_10ep_new_1 \
    --label-dir /data/cuimanni/Composition-1k-testset/alpha_copy \
    --trimap-dir /data/cuimanni/Composition-1k-testset/trimaps 
```

to quick evaluate your inference results.
