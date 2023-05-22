# # initialize accelerator and trackers (if enabled)
from os import makedirs, path, scandir
import pickle
import cv2
import json
from yacs.config import CfgNode as CN
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
import random
import torch
# from src.datasets import FFPP,RPPG
from accelerate import Accelerator
from main import get_config, init_accelerator, set_seed, FFPP
logging.basicConfig(level="DEBUG")


class Obj:
    pass


c = FFPP.get_default_config()
c.augmentation = "normal"
# accelerator =  Accelerator(mixed_precision='no')
# x = FFPP(c,10,2,lambda x: x,accelerator,split="train")

# frames,label,mask,_ = x.get_dict(10,block=True)
# (len(frames),label,len(mask))
print(type(c.dump()))
print(c.dump()[:10])
