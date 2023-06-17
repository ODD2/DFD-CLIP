# # # initialize accelerator and trackers (if enabled)
# from os import makedirs, path, scandir
# import pickle
# import cv2
# import json
# from yacs.config import CfgNode as CN
# from torch.utils.data import Dataset
# from tqdm import tqdm
# import logging
# import random
# import torch
# # from src.datasets import FFPP,RPPG
# from accelerate import Accelerator
# from main import get_config, init_accelerator, set_seed, FFPP
# logging.basicConfig(level="DEBUG")


# class Obj:
#     pass


# c = FFPP.get_default_config()
# c.augmentation = "normal"
# # accelerator =  Accelerator(mixed_precision='no')
# # x = FFPP(c,10,2,lambda x: x,accelerator,split="train")

# # frames,label,mask,_ = x.get_dict(10,block=True)
# # (len(frames),label,len(mask))
# print(type(c.dump()))
# print(c.dump()[:10])
import torchvision

torchvision.io.VideoReader(
    "datasets/dfdc_250/videos/aalscayrfi.avi",
    "video"
)

# import os
# import random
# import torch
# import numpy as np
# from dinov2.models.vision_transformer import DinoVisionTransformer, vit_base

# import os
# import sys
# import random
# import re
# import argparse
# import logging
# from datetime import timedelta, datetime

# import wandb
# import torch
# import numpy as np
# from accelerate import Accelerator, DistributedDataParallelKwargs
# from yacs.config import CfgNode as CN

# from src.models import Detector, CompInvEncoder
# from src.datasets import RPPG, FFPP, DFDC, CDF
# from src.trainer import Trainer, CompInvTrainer
# from src.evaluator import Evaluator, CompInvEvaluator
# from src.callbacks.timer import start_timer, end_timer
# from src.callbacks.metrics import init_metrics, update_metrics, compute_metrics
# from src.callbacks.tracking import update_trackers, cache_best_model
# from src.tools.notify import send_to_telegram

# seed = 10
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True, warn_only=True)

# # m = DinoVisionTransformer(img_size=518, patch_size=14, block_chunks=0)
# m = vit_base(img_size=518, patch_size=14, block_chunks=0, init_values=1.0, ffn_layer="mlp")
# m.load_state_dict(torch.load("misc/dinov2_vitb14_pretrain.pth"))

# # m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',)
# m = m.to("cuda")

# # m.load_state_dict(torch.load("dinov2_vitb14_pretrain.pth"), strict=False)
# x = torch.randn((1, 3, 224, 224), device="cuda", requires_grad=True)
# import warnings
# # warnings.filterwarnings("ignore",".*efficient_attention_forward_cutlass does not have a deterministic implementation.*",UserWarning)
# print("Differences:{}".format(torch.sum((m(x, is_training=True)["q"][0] == m(x, is_training=True)["q"][0]) == False)))
