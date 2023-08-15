# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import logging
import sys
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, IterableDataset
from glob import glob
from math import pi, e
import os
import od.alb
import numpy as np
from PIL import Image
import random
import cv2
import copy
from torch import nn
import sys
import albumentations as alb
from .initialize import init_ff
from inference.datasets import init_cdf
from inference.preprocess import extract_videos_exhaust, extract_videos

import warnings
warnings.filterwarnings('ignore')


class LackOfClipsException(Exception):
    pass


if os.path.isfile('/app/src/utils/library/bi_online_generation.py'):
    sys.path.append('/app/src/utils/library/')
    print('exist library')
    exist_bi = True
else:
    exist_bi = False

# def init_ff(phase,level='frame',n_frames=8):
#     dataset_path='/app/data/FaceForensicsPP/frames/'

#     image_list=[]
#     label_list=[]

#     folder_list = sorted(os.listdir(dataset_path))
#     # folder_list = sorted(glob(dataset_path+'*'))
#     # filelist = []
#     # list_dict = json.load(open(f'/app/data/FaceForensics++/{phase}.json','r'))
#     # for i in list_dict:
#     # 	filelist+=i
#     # folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
#     print('dataset_path')
#     if level =='video':
#         label_list=[0]*len(folder_list)
#         return folder_list,label_list
#     for i in range(len(folder_list)):
#         # images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
#         images_temp=sorted(glob(folder_list[i]+'/*.png'))
#         if n_frames<len(images_temp):
#             images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
#         image_list+=images_temp
#         label_list+=[0]*len(images_temp)

#     return image_list,label_list


class SBI_Dataset(Dataset):
    def __init__(self, phase='train', image_size=224, n_frames=8, raw=True):

        assert phase in ['train', 'val', 'test']

        image_list, label_list = init_ff(phase, raw, 'frame', n_frames=n_frames)
        path_lm = '/landmarks/'
        label_list = [label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace(
            '/frames/', path_lm).replace('.png', '.npy')) and os.path.isfile(image_list[i].replace('/frames/', '/retina/').replace('.png', '.npy'))]
        image_list = [image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace(
            '/frames/', path_lm).replace('.png', '.npy')) and os.path.isfile(image_list[i].replace('/frames/', '/retina/').replace('.png', '.npy'))]
        self.path_lm = path_lm
        print(f'SBI({phase}): {len(image_list)}')

        self.image_list = image_list

        self.image_size = (image_size, image_size)
        self.phase = phase
        self.n_frames = n_frames
        self.raw = raw
        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.blend_ratio = [0.25, 0.5, 0.75, 1, 1, 1]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        flag = True
        while flag:
            try:
                filename = self.image_list[idx]
                img = np.array(Image.open(filename))
                landmark = np.load(filename.replace(
                    '.png', '.npy').replace('/frames/', self.path_lm))[0]
                bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(
                ), landmark[:, 0].max(), landmark[:, 1].max()])
                bboxes = np.load(filename.replace(
                    '.png', '.npy').replace('/frames/', '/retina/'))[:2]
                iou_max = -1
                # multiple face region in a frame dispose.
                for i in range(len(bboxes)):
                    iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
                    if iou_max < iou:
                        bbox = bboxes[i]
                        iou_max = iou
                # distill contour landmark
                landmark = self.reorder_landmark(landmark)

                # do horizontal flip for variation(while training).
                if self.phase == 'train':
                    if np.random.rand() < 0.5:
                        img, _, landmark, bbox = self.hflip(
                            img, None, landmark, bbox)

                # preserve only face region pixels & adjust coordinates in "lm" and "bbox"
                img, landmark, bbox, __ = crop_face(
                    img, landmark, bbox, margin=True, crop_by_bbox=False
                )

                # create self blend images
                img_r, img_f, mask_f = self.self_blending(
                    img.copy(), landmark.copy()
                )

                if self.phase == 'train':
                    transformed = self.transforms(image=img_f.astype(
                        'uint8'), image1=img_r.astype('uint8'))
                    img_f = transformed['image']
                    img_r = transformed['image1']

                img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                    img_f, landmark, bbox, margin=False, crop_by_bbox=True, abs_coord=True, phase=self.phase
                )

                img_r = img_r[y0_new:y1_new, x0_new:x1_new]

                img_f = cv2.resize(
                    img_f, self.image_size, interpolation=cv2.INTER_LINEAR
                ).astype('float32') / 255
                img_r = cv2.resize(
                    img_r, self.image_size, interpolation=cv2.INTER_LINEAR
                ).astype('float32') / 255

                img_f = img_f.transpose((2, 0, 1))
                img_r = img_r.transpose((2, 0, 1))
                flag = False
            except Exception as e:
                print(f"Exception:{e}")
                idx = torch.randint(low=0, high=len(self), size=(1,)).item()

        return img_f, img_r

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=1),
                alb.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
            ], p=1),

            (
                alb.OneOf([
                    RandomDownScale((2, 4) if self.raw else (1, 2), p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ], p=1)
            ),

        ], p=1.)

    def get_transforms(self):
        return alb.Compose([
            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(
                hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
        ],
            additional_targets={f'image1': 'image'},
            p=1.)

    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.01, 0.01), 'y': (-0.01, 0.01)},
            scale=[0.98, 1 / 0.98],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        # only apply elastic deform on mask
        mask = transformed['mask']
        return img, mask

    def self_blending(self, img, landmark):
        H, W = len(img), len(img[0])
        # randomly discard additional face contour coords.
        if np.random.rand() < 0.25:
            landmark = landmark[:68]

        # mask deformation with "bi" library.
        if exist_bi:
            logging.disable(logging.FATAL)
            mask = random_get_hull(landmark, img)[:, :, 0]
            logging.disable(logging.NOTSET)
        else:
            mask = np.zeros_like(img[:, :, 0])  # mask requires only 1 channel.
            # fill white within the hull
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        # generate source, target pair images.
        source = img.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)

        img_blended, mask = B.dynamic_blend(source, img, mask, ratio=np.random.choice(self.blend_ratio))
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img, img_blended, mask

    # distill only contour landmarks.
    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self, img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W - landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W - bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W - bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W - bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W - bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W - bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W - bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new

    def collate_fn(self, batch):
        img_f, img_r = zip(*batch)
        data = {}
        data['img'] = torch.cat([torch.tensor(img_r).float(), torch.tensor(img_f).float()], 0)
        data['label'] = torch.tensor([0] * len(img_r) + [1] * len(img_f))
        return data

    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 4294967295
        np.random.seed(worker_seed)
        random.seed(worker_seed)


class SBI_Temporal_Dataset(SBI_Dataset):
    def __init__(self, phase='train', image_size=224, raw=True, blend_mode="random", clip_size=8, portion=1.0, arduous=True, arduous_portion=1.0, pos_br_base=1.0):
        assert phase in ['train', 'val', 'test']
        assert blend_mode in ["random", "gaussian", "mixed", "both", "all"]
        video_list, _ = init_ff(phase, raw, 'video')
        path_lm = '/landmarks/'
        image_list = []
        self.series = [0]
        self.series_idx = []

        for video_path in video_list:
            video_frames = sorted(glob(video_path + '/*.png'))
            assert len(video_frames) >= clip_size
            clip_pos = 0
            while(clip_pos < len(video_frames)):
                temp_clip_frame = []
                sample_bbox = []
                for i in range(clip_pos, len(video_frames)):
                    image_path = video_frames[i]
                    if (
                        os.path.isfile(image_path.replace('/frames/', path_lm).replace('.png', '.npy')) and
                        os.path.isfile(image_path.replace('/frames/', '/retina/').replace('.png', '.npy'))
                    ):
                        frame_retina_bbox = np.load(image_path.replace(
                            '/frames/', '/retina/').replace('.png', '.npy'))[0][:2]

                        if(len(temp_clip_frame) == 0):
                            sample_bbox = frame_retina_bbox
                        elif(np.linalg.norm(sample_bbox - frame_retina_bbox) > 30):
                            break

                        temp_clip_frame.append(image_path)

                        if(not arduous and len(temp_clip_frame) == clip_size):
                            break
                    else:
                        break
                if(len(temp_clip_frame) >= clip_size):
                    if(arduous):
                        temp_clip_frame = temp_clip_frame[:(
                            clip_size + int((len(temp_clip_frame) - clip_size) * arduous_portion)
                        )]
                    self.series_idx += [len(image_list)]
                    self.series += [len(temp_clip_frame) - clip_size + 1 + self.series[-1]]
                    image_list += temp_clip_frame
                    if(not arduous):
                        break
                clip_pos = i + 1

        self.path_lm = path_lm
        self.image_list = image_list
        self.image_size = (image_size, image_size)
        self.phase = phase
        self.blend_mode = blend_mode
        self.clip_size = clip_size
        self.raw = raw
        self.pos_br_base = pos_br_base
        self.init_alb_param()
        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.extend_transforms = self.get_extend_transforms()

        ### Cut with portion parameter ###
        ignore_data_num = int(len(self.series) * (1 - portion))
        ignore_data_num = -ignore_data_num if ignore_data_num > 0 else None
        self.series = self.series[:ignore_data_num]
        self.series_idx = self.series_idx[:ignore_data_num]
        self.image_list = self.image_list[:self.series_idx[-1] + self.series[-1] - self.series[-2] + self.clip_size + 1]

        print(f'SBI({phase},{blend_mode}): {self.series[-1]}')

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
        return self.series[-1]

    def __getitem__(self, idx):
        flag = True
        flip = False

        if self.phase == 'train':
            if np.random.rand() < 0.5:
                flip = True

        while flag:
            vid_r = []
            vid_f = []
            try:
                for i in range(1, len(self.series), 1):
                    if(idx < self.series[i]):
                        start_idx = self.series_idx[i - 1] + (idx - self.series[i - 1])
                        end_idx = start_idx + self.clip_size
                        break
                # find the bbox for the rest of the frames
                filenames = [
                    self.image_list[i] for i in range(start_idx, end_idx, 1)
                ]

                imgs = [
                    np.array(Image.open(filename))
                    for filename in filenames
                ]

                landmarks = [
                    np.load(filename.replace('.png', '.npy').replace('/frames/', self.path_lm))[0]
                    for filename in filenames
                ]

                #####################################################
                first_frame_dlib_bbox = np.array([
                    landmarks[0][:, 0].min(), landmarks[0][:, 1].min(),
                    landmarks[0][:, 0].max(), landmarks[0][:, 1].max()
                ])

                first_frame_retina_bboxes = np.load(filenames[0].replace(
                    '.png', '.npy').replace('/frames/', '/retina/'))[:, :2]
                sample_bbox = None

                iou_max = -1
                # . multiple face region in a frame dispose.
                for i in range(len(first_frame_retina_bboxes)):
                    iou = IoUfrom2bboxes(first_frame_dlib_bbox, first_frame_retina_bboxes[i].flatten())
                    if iou_max < iou:
                        sample_bbox = first_frame_retina_bboxes[i]
                        iou_max = iou

                if(iou_max == 0):
                    raise Exception("Error, dlib.landmark doesn't intersect with retina bbox")
                ######################################################

                bboxes = [sample_bbox.copy() for _ in range(self.clip_size)]

                # distill contour landmark
                for i in range(self.clip_size):
                    landmarks[i] = self.reorder_landmark(landmarks[i])

                if flip:
                    for i in range(self.clip_size):
                        # do horizontal flip for variation(while training).
                        imgs[i], _, landmarks[i], bboxes[i] = self.hflip(imgs[i], None, landmarks[i], bboxes[i])

                # preserve only face region pixels /w margin & adjust coordinates in "lm" and "bbox"
                for i in range(self.clip_size):
                    imgs[i], landmarks[i], bboxes[i], __ = crop_face(
                        imgs[i], landmarks[i], bboxes[i], margin=True, crop_by_bbox=True
                    )

                # create self blend images
                _, imgs_f, _, blend_weights, blend_ratio = self.self_blending(
                    copy.deepcopy(imgs), copy.deepcopy(landmarks)
                )

                imgs_r = copy.deepcopy(imgs)

                if(self.phase == "train"):
                    crop_seed = np.random.randint(0, 123456)
                else:
                    crop_seed = None
                # crop face with bbox only
                for i in range(self.clip_size):
                    img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                        imgs_f[i],
                        landmarks[i],
                        bboxes[i],
                        margin=False,
                        crop_by_bbox=True,
                        abs_coord=True,
                        phase="train",
                        seed=crop_seed
                    )
                    img_r = imgs_r[i][y0_new:y1_new, x0_new:x1_new]

                    vid_r.append(img_r)
                    vid_f.append(img_f)
                # video level transform
                vid_l = [vid_r, vid_f]
                ext_blend_ratio = 0
                if self.phase == "train":
                    # extensional augmentation for real images.
                    if np.random.rand() < 0.5:
                        ext_blend_ratio = np.random.choice(self.blend_random_ratio)
                       # general video augementation
                        vid_img = {
                            f"image{i}": img_i.astype('uint8')
                            for i, img_i in enumerate(vid_r)
                        }
                        transformed = self.extend_transforms(
                            image=vid_r[0].astype('uint8'),
                            **vid_img
                        )
                        for i in range(self.clip_size):
                            vid_l[0][i] = transformed[f'image{i}'] * ext_blend_ratio + (1 - ext_blend_ratio) * vid_r[i]

                    # general video augementation
                    vid_img = {
                        f"image{t}_{i}": img_i.astype('uint8')
                        for t, vid_t in enumerate(vid_l)
                        for i, img_i in enumerate(vid_t)
                    }
                    transformed = self.transforms(
                        image=vid_r[0].astype('uint8'),
                        **vid_img
                    )
                    for t in range(2):
                        for i in range(self.clip_size):
                            vid_l[t][i] = transformed[f'image{t}_{i}']
                # resize and normalize
                for t in range(2):
                    for i in range(self.clip_size):
                        vid_l[t][i] = cv2.resize(
                            vid_l[t][i],
                            self.image_size,
                            interpolation=cv2.INTER_LINEAR
                        ).astype('float32') / 255
                        vid_l[t][i] = vid_l[t][i].transpose((2, 0, 1))

                # done, set flag
                flag = False
            except Exception as e:
                print(e)
                idx = torch.randint(low=0, high=len(self), size=(1,)).item()

        return vid_f, vid_r, blend_weights, blend_ratio, ext_blend_ratio

    def self_blending(self, imgs, landmarks):
        # randomly discard additional face contour coords.
        if np.random.rand() < 0.25:
            landmarks = [landmarks[i][:68] for i in range(self.clip_size)]
        # mask deformation with "bi" library.
        if exist_bi:
            logging.disable(logging.FATAL)
            hull_type = np.random.choice([0, 1, 2, 3])
            masks = [
                random_get_hull(landmarks[i], imgs[i], hull_type=hull_type)[:, :, 0]
                for i in range(self.clip_size)
            ]
            logging.disable(logging.NOTSET)
        else:
            # mask requires only 1 channel.
            masks = [np.zeros_like(imgs[i][:, :, 0]) for i in range(self.clip_size)]
            for i in range(self.clip_size):
                # fill white within the hull
                cv2.fillConvexPoly(masks[i], cv2.convexHull(landmarks[i]), 1.)

        sources = copy.deepcopy(imgs)

        results = self.source_transforms(
            image=imgs[0].astype(np.uint8),
            **{
                f"image{i}": imgs[i].astype(np.uint8) for i in range(self.clip_size)
            }
        )

        _imgs = [results[f"image{i}"] for i in range(self.clip_size)]
        if np.random.rand() < 0.5:
            sources = _imgs
        else:
            imgs = _imgs
        sources, masks = self.randaffine(sources, masks)
        if(self.blend_mode == "random"):
            ratios = self.blend_random()
        elif(self.blend_mode == "gaussian"):
            ratios = self.blend_gaussian()
        elif(self.blend_mode == "mixed"):
            ratios = self.blend_mixed()
        elif(self.blend_mode == "both"):
            ratios = np.random.choice([self.blend_mixed, self.blend_random])()
        elif(self.blend_mode == "all"):
            ratios = np.random.choice([self.blend_mixed, self.blend_random, self.blend_hazzard])()
        else:
            ratios = [1 for _ in range(self.clip_size)]
        blended_imgs = [None for _ in range(self.clip_size)]
        for i in range(self.clip_size):
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
            self.gaussian_pdf((i / (self.clip_size - 1) * 4) - 2 + _s, _o)
            for i in range(self.clip_size)
        ]
        return ratios

    def blend_random(self):
        _ratio = np.random.choice(self.blend_random_ratio)
        ratios = [
            _ratio for _ in range(self.clip_size)
        ]
        return ratios

    def blend_hazzard(self):
        _ratio = np.random.choice(self.blend_random_ratio)
        ratios = [
            _ratio * np.random.rand() for _ in range(self.clip_size)
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
                **{f'image{i}': 'image' for i in range(0, self.clip_size, 1)},
                **{f'mask{i}': 'mask' for i in range(0, self.clip_size, 1)}
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
                **{f'image{i}': 'image' for i in range(0, self.clip_size, 1)},
                **{f'mask{i}': 'mask' for i in range(0, self.clip_size, 1)}
            },
            p=1.
        )

        transformed = f(
            image=imgs[0],
            **{f"image{i}": imgs[i] for i in range(self.clip_size)},
            mask=masks[0],
            **{f"mask{i}": masks[i] for i in range(self.clip_size)},
        )

        imgs = [transformed[f'image{i}'] for i in range(self.clip_size)]

        masks = [transformed[f'mask{i}'] for i in range(self.clip_size)]

        transformed = g(
            image=imgs[0],
            **{f"image{i}": imgs[i] for i in range(self.clip_size)},
            mask=masks[0],
            **{f"mask{i}": masks[i] for i in range(self.clip_size)},
        )

        # only apply elastic deform on mask
        masks = [transformed[f'mask{i}'] for i in range(self.clip_size)]

        return imgs, masks

    def get_transforms(self):
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
                f'image{t}_{i}': 'image' for t in range(0, 2) for i in range(0, self.clip_size, 1)
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
                f'image{i}': 'image' for i in range(0, self.clip_size, 1)
            },
            p=1.0
        )

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose(
                [
                    od.alb.MarginalRGBShift(
                        self.src_alb_rgb_bound,
                        self.src_alb_rgb_bound,
                        self.src_alb_rgb_bound,
                        self.src_alb_rgb_margin,
                        p=1
                    ),
                    od.alb.MarginalHueSaturationValue(
                        self.src_alb_hue_bound,
                        self.src_alb_hue_bound,
                        self.src_alb_hue_bound,
                        self.src_alb_hue_margin,
                        p=1
                    ),
                    od.alb.MarginalRandomBrightnessContrast(
                        self.src_alb_bright_bound,
                        self.src_alb_bright_bound,
                        self.src_alb_bright_margin,
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
                f'image{i}': 'image' for i in range(0, self.clip_size)
        },
            p=1.
        )

    def collate_fn(self, batch):
        vid_f, vid_r, bw_f, br, ext_br = zip(*batch)
        bw_r = [1] + [0 for _ in range(self.clip_size)]
        data_seq = []
        label_seq = []
        prob_seq = []
        weight_seq = []
        for i in range(len(vid_f)):
            data_seq.append(torch.tensor(vid_f[i]).float().unsqueeze(0))
            weight_seq.append(torch.tensor([0] + bw_f[i]).float().unsqueeze(0))
            label_seq.append(1)
            _br = min(self.pos_br_base + (1 - self.pos_br_base) * br[i], 1)
            prob_seq.append([1 - _br, _br])

            data_seq.append(torch.tensor(vid_r[i]).float().unsqueeze(0))
            weight_seq.append(torch.tensor(bw_r).float().unsqueeze(0))
            label_seq.append(0)
            _ext_br = 1
            prob_seq.append([_ext_br, 1 - _ext_br])
        data = {}
        data['img'] = torch.cat(data_seq, 0)
        data['weight'] = torch.cat(weight_seq, 0)
        data['label'] = torch.tensor(label_seq)
        data['prob'] = torch.tensor(prob_seq)
        return data

    def gaussian_pdf(self, x, o):
        return pow(e, (-0.5 * pow((x / o), 2)))


class CDF_Test_Dataset(Dataset):
    def __init__(self, image_size=224, n_clips=8, clip_size=8, portion=1.0):
        # sampling decisions
        video_list, target_list = init_cdf()
        self.image_size = image_size
        self.n_clips = n_clips
        self.clip_size = clip_size
        sample_num = int(len(target_list) * portion)
        sample_idx = np.random.choice(range(len(target_list)), sample_num, replace=False)
        self.video_list = [video_list[i] for i in sample_idx]
        self.target_list = [target_list[i] for i in sample_idx]
        print(f'CDF(test): {len(self.target_list)}')

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        try:
            filename = self.video_list[index]
            _clips = extract_videos(
                filename, self.n_clips, self.clip_size, None, (self.image_size, self.image_size)
            )
            if(len(_clips) < self.n_clips):
                raise LackOfClipsException("CDF Test Dataset: Insufficient Clips")

            _labels = torch.tensor([self.target_list[index]] * len(_clips))

            _clips = torch.tensor(_clips).float() / 255

            return _clips, _labels

        except LackOfClipsException as e:
            print(e)
            return self.__getitem__(np.random.randint(0, self.__len__()))

        except Exception as e:
            raise(e)

    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 4294967295
        np.random.seed(worker_seed)
        random.seed(worker_seed)


# if __name__ == '__main__':
#     import blend as B
#     from initialize import *
#     from funcs import IoUfrom2bboxes, crop_face, RandomDownScale
#     if exist_bi:
#         from library.bi_online_generation import random_get_hull
#     seed = 10
#     random.seed(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     image_dataset = SBI_Dataset(phase='test', image_size=256)
#     batch_size = 64
#     dataloader = torch.utils.data.DataLoader(image_dataset,
#                                              batch_size=batch_size,
#                                              shuffle=True,
#                                              collate_fn=image_dataset.collate_fn,
#                                              num_workers=0,
#                                              worker_init_fn=image_dataset.worker_init_fn
#                                              )
#     data_iter = iter(dataloader)
#     data = next(data_iter)
#     img = data['img']
#     img = img.view((-1, 3, 256, 256))
#     utils.save_image(img, 'loader.png', nrow=batch_size,
#                      normalize=False, range=(0, 1))
# else:
#     from utils import blend as B
#     from .initialize import *
#     from .funcs import IoUfrom2bboxes, crop_face, RandomDownScale
#     if exist_bi:
#         from utils.library.bi_online_generation import random_get_hull


if __name__ == '__main__':
    import utils.blend as B
    from utils.initialize import *
    from utils.funcs import IoUfrom2bboxes, crop_face, RandomDownScale
    if exist_bi:
        from utils.library.bi_online_generation import random_get_hull
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = SBI_Temporal_Dataset(
        phase="train",
        image_size=380,
        raw=True,
        clip_size=8,
        portion=0.1,
        blend_mode="all",
        arduous=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
        collate_fn=train_dataset.collate_fn
    )
    it = iter(train_loader)
    data = next(it)
else:
    from utils import blend as B
    from .initialize import *
    from .funcs import IoUfrom2bboxes, crop_face, RandomDownScale
    if exist_bi:
        from utils.library.bi_online_generation import random_get_hull
