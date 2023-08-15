
import os
import cv2
import torch
import pickle
import logging
import random
import torchvision
from tqdm import tqdm
from accelerate import Accelerator
import numpy as np
import albumentations as alb
from os import path, scandir, makedirs
from src.sbi import blend as B
from src.sbi.initialize import *
from src.sbi.funcs import IoUfrom2bboxes, crop_face, RandomDownScale
from src.sbi.library.bi_online_generation import random_get_hull
from torch.utils.data import Dataset, default_collate
from yacs.config import CfgNode as CN
from src.datasets import FFPP
import copy


class SBI(FFPP):
    @staticmethod
    def get_default_config():
        C = CN()
        C.category = 'train'
        C.root_dir = './datasets/ffpp/'
        C.vid_ext = ".avi"
        C.compressions = ['c23']
        C.name = "SBI"
        C.scale = 1.0
        C.pack = False
        C.pair = False
        C.contrast = False
        return C

    @staticmethod
    def validate_config(config):
        config = config.clone()
        config.defrost()

        assert type(config.category) == str
        assert len(config.category) > 0

        assert type(config.root_dir) == str
        assert len(config.root_dir) > 0

        assert type(config.vid_ext) == str
        assert len(config.vid_ext) > 0

        assert type(config.compressions) == list
        assert len(config.compressions) > 0

        assert type(config.scale) == float
        assert 0 < config.scale <= 1

        config.pair = False
        config.contrast = True
        config.pack = False
        config.types = ["REAL"]

        config.freeze()
        return config

    def __init__(self, config, num_frames, clip_duration, transform, accelerator, split='train', n_px=224, index=0):
        self.TYPE_DIRS = {
            'REAL': 'real/',
            'DF': 'DF/',
            'FS': 'FS/',
            'F2F': 'F2F/',
            'NT': 'NT/',
            'FSh': 'FSh/'
        }
        config = self.validate_config(config)

        self.category = config.category.lower()
        self.name = config.name.lower()
        self.root = path.expanduser(config.root_dir)
        self.vid_ext = config.vid_ext
        self.types = sorted(set(config.types), reverse=True)
        self.compressions = sorted(set(config.compressions), reverse=True)
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.split = split
        self.transform = transform
        self.n_px = n_px
        self.pack = config.pack
        self.pair = config.pair
        self.contrast = config.contrast

        self.index = index
        self.scale = config.scale
        # video info list
        self.video_list = []

        # record missing videos in the csv file for further usage.
        self.stray_videos = {}

        # stacking data clips
        self.stack_video_clips = []

        self.transform = transform
        self.init_alb_param()
        self.general_transoforms = self.get_general_transforms()
        self.source_transforms = self.get_source_transforms()

        self._build_video_table(accelerator)
        self._build_video_list(accelerator)

        if split == "train":
            augmentations = [
                alb.RandomResizedCrop(
                    self.n_px, self.n_px, scale=(0.4, 1.0), ratio=(1, 1), p=1.0
                ),
                alb.HorizontalFlip()
            ]

            self.sequence_augmentation = alb.ReplayCompose(
                augmentations,
                p=1.
            )
            self.frame_augmentation = None

            def driver(x, replay={}):
                # transform to numpy, the alb required format
                x = [_x.numpy().transpose((1, 2, 0)) for _x in x]
                # frame augmentation
                if (not self.frame_augmentation == None):
                    if ("frame" in replay):
                        assert len(replay["frame"]) == len(x), "Error! frame replay should match the number of frames"
                        x = [
                            alb.ReplayCompose.replay(_r, image=_x)["image"] for _x, _r in zip(x, replay["frame"])
                        ]

                    else:
                        replay["frame"] = [None for _ in x]
                        for i, _x in enumerate(x):
                            result = self.frame_augmentation(image=_x)
                            x[i] = result["image"]
                            replay["frame"][i] = result["replay"]
                # sequence augmentation
                if (not self.sequence_augmentation == None):
                    if ("video" in replay):
                        x = [alb.ReplayCompose.replay(replay["video"], image=_x)[
                            "image"] for _x in x]
                    else:
                        replay["video"] = self.sequence_augmentation(image=x[0])[
                            "replay"]
                        x = [alb.ReplayCompose.replay(replay["video"], image=_x)[
                            "image"] for _x in x]
                # revert to tensor
                x = [torch.from_numpy(_x.transpose((2, 0, 1))) for _x in x]
                return x, replay
        else:
            def driver(x, replay={}):
                return x, replay
        self.augmentation = driver

    def init_alb_param(self):
        # Albumentation Strength Parameters
        # - Source Transforms
        self.src_alb_rgb_bound = (-20, 20)
        self.src_alb_rgb_margin = 5
        self.src_alb_hue_bound = (-0.3, 0.3)
        self.src_alb_hue_margin = 0.1
        self.src_alb_bright_bound = (-0.1, 0.1)
        self.src_alb_bright_margin = 0.02
        self.src_alb_dscale_bound = (1, 3)
        self.src_alb_sharpen_params = {
            "alpha": (0.1, 0.25),
            "lightness": (0.25, 0.5)
        }
        # - Affine Transforms
        self.aff_alb_aff_params = {
            "translate_percent": {'x': (-0.02, 0.02), 'y': (-0.015, 0.015)},
            "scale": [0.95, 1 / 0.95],
            "fit_output": False
        }
        self.aff_alb_els_params = {
            "alpha": 50,
            "sigma": 7,
            "alpha_affine": 0,
        }
        # - General Transforms (Train Only)
        self.gen_alb_rgb_bound = (-20, 20)
        self.gen_alb_hue_bound = (-0.3, 0.3)
        self.gen_alb_bright_bound = (-0.3, 0.3)
        self.gen_alb_dscale_bound = (1, 3)
        self.gen_alb_compress_params = {
            "quality_lower": 40,
            "quality_upper": 70,
        }
        # - Blending Setups
        self.blend_random_ratio = (
            [0.2, 0.4, 0.6, 0.8, 1, 1, 1]
        )
        self.blend_gauss_variance = (
            [3.0, 5.0, 7.0, 100, 100]
        )

    def __len__(self):
        return self.stack_video_clips[-1]

    def __getitem__(self, idx):
        result = self.get_dict(idx, False)
        return *[[_r[name] for _r in result] for name in ["frames", "label", "mask", "speed"]], [self.index] * 2

    def get_dict(self, idx, block=False):
        while (True):
            try:
                # video_idx =  next(i for i,x in enumerate(self.stack_video_clips) if  idx < x)
                # df_type, comp, video_name, clips = self.video_list[video_idx]
                video_idx, df_type, comp, video_name, clips = self.video_info(idx)
                video_meta = self.video_table[df_type][comp][video_name]
                video_offset_duration = (
                    idx - (0 if video_idx == 0 else self.stack_video_clips[video_idx - 1])
                ) * self.clip_duration
                logging.debug(f"Item/Video Index:{idx}/{video_idx}")
                logging.debug(f"Item DF/COMP:{df_type}/{comp}")

                if (self.split == "train"):
                    # the slow motion factor for video data augmentation
                    video_speed_factor = random.random() * 0.5 + 0.5
                    video_shift_factor = random.random() * (1 - video_speed_factor)
                else:
                    video_speed_factor = 1
                    video_shift_factor = 0

                logging.debug(f"Video Speed Motion Factor: {video_speed_factor}")
                logging.debug(f"Video Shift Factor: {video_shift_factor}")

                # video frame processing
                _frames = []
                _landmarks = []

                vid_path = video_meta["path"]

                vid_reader = torchvision.io.VideoReader(
                    vid_path,
                    "video"
                )
                # - frames per second
                video_sample_freq = vid_reader.get_metadata()["video"]["fps"][0]
                # - the amount of frames to skip
                video_sample_offset = int(
                    video_offset_duration + self.clip_duration * video_shift_factor
                )
                # - the amount of frames for the duration of a clip
                video_clip_samples = int(
                    video_sample_freq * self.clip_duration * video_speed_factor
                )
                # - the amount of frames to skip in order to meet the num_frames per clip.(excluding the head & tail frames )
                if (self.num_frames == 1):
                    video_sample_stride = 0
                else:
                    video_sample_stride = (
                        (video_clip_samples - 1) / (self.num_frames - 1)
                    ) / video_sample_freq
                logging.debug(f"Loading Video: {vid_path}")
                logging.debug(f"Sample Offset: {video_sample_offset}")
                logging.debug(f"Sample Stride: {video_sample_stride}")
                # load landmark in advance
                vid_lm_path = vid_path.replace("videos", "landmarks").replace(self.vid_ext, ".npy")
                logging.debug(f"Loading Landmark: {vid_lm_path}")
                vid_landmarks = np.load(vid_lm_path)
                # - fetch frames of clip duration
                for sample_idx in range(self.num_frames):
                    try:
                        _time = video_sample_offset + sample_idx * video_sample_stride
                        vid_reader.seek(_time)
                        frame = next(vid_reader)
                        _frames.append(frame["data"])
                        _landmarks.append(vid_landmarks[int(_time*video_sample_freq)])
                    except Exception as e:
                        raise Exception(f"unable to read video frame of sample index:{sample_idx}")
                del vid_reader
                _frames = [_x.numpy().transpose((1, 2, 0)) for _x in _frames]
                _, imgs_f, _, blend_weights, blend_ratio = self.self_blending(
                    copy.deepcopy(_frames), copy.deepcopy(_landmarks)
                )
                imgs_r = copy.deepcopy(_frames)

                # transform for real clip during training:
                if (self.split == "train"):
                    transformed = self.general_transoforms(
                        image=imgs_r[0].astype('uint8'),
                        **{
                            f"image{i}": imgs_r[i].astype('uint8')
                            for i in range(self.num_frames)
                        }
                    )
                    for i in range(self.num_frames):
                        imgs_r[i] = transformed[f'image{i}']

                # revert to tensor
                imgs_r = [torch.from_numpy(_x.transpose((2, 0, 1))) for _x in imgs_r]
                imgs_f = [torch.from_numpy(_x.transpose((2, 0, 1))) for _x in imgs_f]

                # perform general size augmentations
                imgs_r, replay = self.augmentation(imgs_r, {})
                imgs_f, _ = self.augmentation(imgs_f, replay)

                # stack list of torch frames to tensor
                imgs_r = torch.stack(imgs_r)
                imgs_f = torch.stack(imgs_f)

                results = []
                for clip, label in zip([imgs_r, imgs_f], [0, 1]):
                    frames = {}
                    # transformation
                    if (self.transform):
                        clip = self.transform(clip)
                    frames[comp] = clip
                    logging.debug(f"Video: SBI {vid_path} for {'FAKE' if label else 'REAL'} , Completed!")
                    # padding and masking missing frames.
                    mask = torch.tensor([1.] * len(frames[comp]) +
                                        [0.] * (self.num_frames - len(frames[comp])), dtype=torch.bool)

                    if len(frames[comp]) < self.num_frames:
                        diff = self.num_frames - len(frames[comp])
                        padding = torch.zeros((diff, *frames[comp].shape[1:]), dtype=torch.uint8)
                        frames[comp] = torch.concatenate((frames[comp], padding))

                    results.append({
                        "frames": frames,
                        "label": label,
                        "mask": mask,
                        "speed": video_speed_factor,
                        "idx": idx
                    })

                return results
            except Exception as e:
                logging.error(f"Error occur: {e}")
                if block:
                    raise e
                else:
                    idx = random.randrange(0, len(self))

    def self_blending(self, imgs, landmarks):
        # randomly discard additional face contour coords.
        if np.random.rand() < 0.25:
            landmarks = [landmarks[i][:68] for i in range(self.num_frames)]
        # mask deformation with "bi" library.
        logging.disable(logging.FATAL)
        hull_type = np.random.choice([0, 1, 2, 3])
        masks = [
            random_get_hull(landmarks[i], imgs[i], hull_type=hull_type)[:, :, 0]
            for i in range(self.num_frames)
        ]
        logging.disable(logging.NOTSET)

        sources = copy.deepcopy(imgs)

        results = self.source_transforms(
            image=imgs[0].astype(np.uint8),
            **{
                f"image{i}": imgs[i].astype(np.uint8) for i in range(self.num_frames)
            }
        )

        _imgs = [results[f"image{i}"] for i in range(self.num_frames)]
        if np.random.rand() < 0.5:
            sources = _imgs
        else:
            imgs = _imgs
        sources, masks = self.randaffine(sources, masks)
        ratios = self.blend_random()
        blended_imgs = [None for _ in range(self.num_frames)]
        for i in range(self.num_frames):
            blended_imgs[i], masks[i] = B.dynamic_blend(sources[i], imgs[i], masks[i], ratio=ratios[i])
            blended_imgs[i] = blended_imgs[i].astype(np.uint8)
            imgs[i] = imgs[i].astype(np.uint8)
        blend_weights = torch.tensor(ratios).float()
        blend_weights = (blend_weights / torch.sum(blend_weights, dim=-1)).tolist()
        return imgs, blended_imgs, masks, blend_weights, max(ratios)

    def blend_mixed(self):
        _r = np.random.choice(self.blend_random_ratio)
        ratios = [
            _r * i for i in self.blend_gaussian()
        ]
        return ratios

    def blend_gaussian(self):
        _o = np.random.choice(self.blend_gauss_variance)
        _s = np.random.randint(-2, 3)
        ratios = [
            self.gaussian_pdf((i / (self.num_frames - 1) * 4) - 2 + _s, _o)
            for i in range(self.num_frames)
        ]
        return ratios

    def blend_random(self):
        _ratio = np.random.choice(self.blend_random_ratio)
        ratios = [
            _ratio for _ in range(self.num_frames)
        ]
        return ratios

    def blend_hazzard(self):
        _ratio = np.random.choice(self.blend_random_ratio)
        ratios = [
            _ratio * np.random.rand() for _ in range(self.num_frames)
        ]
        return ratios

    def randaffine(self, imgs, masks):
        f = alb.Compose(
            [
                alb.Affine(
                    **self.aff_alb_aff_params,
                    p=1
                )
            ],
            additional_targets={
                **{f'image{i}': 'image' for i in range(0, self.num_frames, 1)},
                **{f'mask{i}': 'mask' for i in range(0, self.num_frames, 1)}
            },
            p=1.
        )

        g = alb.Compose(
            [
                alb.ElasticTransform(
                    **self.aff_alb_els_params,
                    p=1
                )
            ],
            additional_targets={
                **{f'image{i}': 'image' for i in range(0, self.num_frames, 1)},
                **{f'mask{i}': 'mask' for i in range(0, self.num_frames, 1)}
            },
            p=1.
        )

        transformed = f(
            image=imgs[0],
            **{f"image{i}": imgs[i] for i in range(self.num_frames)},
            mask=masks[0],
            **{f"mask{i}": masks[i] for i in range(self.num_frames)},
        )

        imgs = [transformed[f'image{i}'] for i in range(self.num_frames)]

        masks = [transformed[f'mask{i}'] for i in range(self.num_frames)]

        transformed = g(
            image=imgs[0],
            **{f"image{i}": imgs[i] for i in range(self.num_frames)},
            mask=masks[0],
            **{f"mask{i}": masks[i] for i in range(self.num_frames)},
        )

        # only apply elastic deform on mask
        masks = [transformed[f'mask{i}'] for i in range(self.num_frames)]

        return imgs, masks

    def get_general_transforms(self):
        return alb.Compose(
            [
                alb.RGBShift(
                    self.gen_alb_rgb_bound,
                    self.gen_alb_rgb_bound,
                    self.gen_alb_rgb_bound,
                    p=0.5
                ),
                alb.HueSaturationValue(
                    self.gen_alb_hue_bound,
                    self.gen_alb_hue_bound,
                    self.gen_alb_hue_bound,
                    p=0.5
                ),
                alb.RandomBrightnessContrast(
                    self.gen_alb_bright_bound,
                    self.gen_alb_bright_bound,
                    p=0.5
                ),
                RandomDownScale(
                    self.gen_alb_dscale_bound,
                    p=0.5
                ),
                alb.ImageCompression(
                    **self.gen_alb_compress_params,
                    p=0.5
                ),
            ],
            additional_targets={
                f'image{t}_{i}': 'image' for t in range(0, 2) for i in range(0, self.num_frames, 1)
            },
            p=1.0
        )

    def get_extend_transforms(self):
        return alb.Compose(
            alb.OneOf(
                [
                    alb.GaussNoise(
                        var_limit=(100, 200),
                        mean=0,
                        per_channel=True,
                        p=1.0
                    ),
                    alb.ImageCompression(
                        quality_lower=20,
                        quality_upper=50,
                        p=1.0
                    ),
                    alb.MultiplicativeNoise(
                        multiplier=(0.9, 1.1),
                        per_channel=False,
                        elementwise=True,
                        always_apply=False,
                        p=1.0
                    ),
                ],
                p=1.0
            ),
            additional_targets={
                f'image{i}': 'image' for i in range(0, self.num_frames, 1)
            },
            p=1.0
        )

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose(
                [
                    alb.RGBShift(
                        self.src_alb_rgb_bound,
                        self.src_alb_rgb_bound,
                        self.src_alb_rgb_bound,
                        p=1
                    ),
                    alb.HueSaturationValue(
                        self.src_alb_hue_bound,
                        self.src_alb_hue_bound,
                        self.src_alb_hue_bound,
                        p=1
                    ),
                    alb.RandomBrightnessContrast(
                        self.src_alb_bright_bound,
                        self.src_alb_bright_bound,
                        p=1
                    ),
                ],
                p=1
            ),
            (
                alb.OneOf(
                    [
                        RandomDownScale(self.src_alb_dscale_bound, p=1),
                        alb.Sharpen(**self.src_alb_sharpen_params, p=1),
                    ],
                    p=1
                )
            ),
        ],
            additional_targets={
                f'image{i}': 'image' for i in range(0, self.num_frames)
        },
            p=1.
        )

    def gaussian_pdf(self, x, o):
        return pow(e, (-0.5 * pow((x / o), 2)))
