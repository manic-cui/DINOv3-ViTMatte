import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from os.path import join as opj
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
cfg=LazyConfig.load('configs/ViTMatte_S_100ep.py')
model=instantiate(cfg.model)
DetectionCheckpointer(model).load('/data/cuimanni/vitmatte_result/output_of_train/ViTMatte_S_10ep_10/model_final.pth')
print(model)