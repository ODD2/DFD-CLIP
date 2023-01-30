import argparse
from os import path, makedirs
from sys import stderr
from glob import glob
import math
import re

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator

from facexlib.detection import init_detection_model
from facexlib.alignment import init_alignment_model, landmark_98_to_68

class VideoDataset(Dataset):
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.video_paths = glob(path.join(data_dir, '**/*.mp4'), recursive=True)
    def __len__(self):
        return len(self.video_paths)
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        # remove data_dir from video_path
        video_name = video_path[len(self.data_dir):]
        lm_save_name = path.join(self.save_dir, path.splitext(video_name)[0]) + '.npy'
        lm_save_name = re.sub('/videos/', '/landmarks/', lm_save_name)

        # decode video into frames
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        return video_name, lm_save_name, frames, fps

def collate_fn(batch):
    # just forward the dataset
    # only work with batch_size=1
    return batch[0]

@torch.no_grad()
def main(args):
    accelerator = Accelerator()

    # expand ~ and ensure the final slash
    data_dir = path.join(path.expanduser(args.data_dir), '')
    save_dir = path.join(path.expanduser(args.save_dir), '')

    video_dataset = VideoDataset(data_dir, save_dir)
    video_dataloader = accelerator.prepare(DataLoader(video_dataset, collate_fn=collate_fn))

    with accelerator.main_process_first():
        retinaface = init_detection_model(args.detection_model_name, device=accelerator.device)
        align_net = init_alignment_model(args.alignment_model_name, device=accelerator.device)

    progress_bar = tqdm(video_dataloader, disable=not accelerator.is_local_main_process)
    for video_name, lm_save_name, frames, fps in progress_bar:
        if fps < 1:
            print(f'Invalid fps of {fps} in {video_name}')
            continue

        progress_bar.set_description(video_name)
        # batch detect faces
        bboxes_in_frames = []
        for i in range(0, len(frames), args.batch_size):
            batch = torch.from_numpy(np.stack(frames[i:i+args.batch_size]))
            bboxes_batch, _ = retinaface.batched_detect_faces(batch, args.conf_threshold)
            bboxes_in_frames.extend(bboxes_batch)

        assert len(bboxes_in_frames) == len(frames)
        h, w, _ = frames[0].shape
        last_bboxes = np.array([[0, 0, w, h, 1]])
        landmarks_list = []
        for frame, bboxes in zip(frames, bboxes_in_frames):
            if len(bboxes) == 0:
                bboxes = last_bboxes
            else:
                last_bboxes = bboxes

            # keep the salient bbox (face) only
            x1, y1, x2, y2 = bboxes[0][:4]

            # apply padding to bbox
            padding = min(x1,y1,
                          frames[0].shape[1] - x2,
                          frames[0].shape[0] - y2,
                          int(args.max_bbox_padding * (x2-x1)))
            x1 = int(x1 - padding)
            x2 = int(x2 + padding)
            y1 = int(y1 - padding)
            y2 = int(y2 + padding)

            face = frame[y1:y2, x1:x2]
            landmarks = align_net.get_landmarks(face, device=accelerator.device)
            landmarks = landmark_98_to_68(landmarks)
            landmarks[:, 0] += x1
            landmarks[:, 1] += y1
            landmarks_list.append(landmarks)

        makedirs(path.dirname(lm_save_name), exist_ok=True)
        np.save(lm_save_name, np.stack(landmarks_list))

def get_argparser():
    parser = argparse.ArgumentParser(description='Face detector')
    parser.add_argument("--data_dir", required=True, type=str, help="dataset root directory")
    parser.add_argument("--save_dir", default='', type=str, help="output root directory")
    parser.add_argument(
        '--detection_model_name', type=str, default='retinaface_mobile0.25', help='retinaface_resnet50 | retinaface_mobile0.25')
    parser.add_argument(
        '--alignment_model_name', type=str, default='awing_fan')
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--conf_threshold", default=0.9, type=float, help="face confidence threshold")
    parser.add_argument("--max_bbox_padding", default=0.15, type=float, help="maximum amount of padding of bounding box")
    parser.add_argument("--dce_scale_factor", default=12, type=int, help="Scale factor to speed up low-light enhancing network")
    parser.add_argument("--save_as", default="videos", type=str, help="frames, videos")
    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    args.save_dir = args.save_dir if args.save_dir else args.data_dir

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)
