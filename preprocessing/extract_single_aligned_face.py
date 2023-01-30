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


def affine_transform(frame, landmarks, reference, grayscale=False, target_size=(256, 256),
                     reference_size=(256, 256), stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
                     interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                     border_value=0):
    # Prepare everything
    if grayscale and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stable_reference = np.vstack([reference[x] for x in stable_points])
    stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
    stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

    # Warp the face patch and the landmarks
    transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                            stable_reference, method=cv2.LMEDS)[0]
    transformed_frame = cv2.warpAffine(frame, transform, dsize=(target_size[0], target_size[1]),
                                flags=interpolation, borderMode=border_mode, borderValue=border_value)
    transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

    return transformed_frame, transformed_landmarks


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def crop_patch(frames, landmarks, reference, args):
    sequence = []
    length = min(len(landmarks), len(frames))
    for frame_idx in range(length):
        frame = frames[frame_idx]
        window_margin = min(args.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
        smoothed_landmarks = np.mean(
            [landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0
        )
        smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
        transformed_frame, transformed_landmarks = affine_transform(
            frame, smoothed_landmarks, reference, grayscale=False
        )
        sequence.append(
            cut_patch(
                transformed_frame,
                transformed_landmarks[args.start_idx:args.stop_idx],
                args.crop_height//2,
                args.crop_width//2,
            )
        )
    return np.array(sequence)


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
        video_save_dir = path.join(self.save_dir, path.splitext(video_name)[0])

        if path.isfile(path.join(video_save_dir, 'done')):
            print(f'skipping completed video: {video_name}', file=stderr)
            return [None] * 5

        # decode video into frames
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps < 1:
            print(f'Invalid fps of {fps} in {video_name}')
            return [None] * 5

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        landmarks_path = re.sub('/videos/', '/landmarks/', video_path)
        landmarks_path = re.sub('\\.mp4', '.npy', landmarks_path)
        landmarks = np.load(landmarks_path)

        return video_name, video_save_dir, frames, fps, landmarks

def collate_fn(batch):
    # just forward the dataset
    # only work with batch_size=1
    return batch[0]

@torch.no_grad()
def main(args):
    accelerator = Accelerator()
    reference = np.load(args.mean_face)

    # expand ~ and ensure the final slash
    data_dir = path.join(path.expanduser(args.data_dir), '')
    save_dir = path.join(path.expanduser(args.save_dir), '')

    video_dataset = VideoDataset(data_dir, save_dir)
    video_dataloader = accelerator.prepare(DataLoader(video_dataset, collate_fn=collate_fn))

    progress_bar = tqdm(video_dataloader, disable=not accelerator.is_local_main_process)
    for video_name, video_save_dir, frames, fps, landmarks in progress_bar:
        if video_name is None:
            # completed video
            # see the dataset
            continue

        progress_bar.set_description(video_name)
        cropped_frames = crop_patch(frames, landmarks, reference, args)

        # write to a lossless images every 1 second
        fname_pad = math.floor(math.log(len(cropped_frames) / fps, 10)) + 1
        for i in range(0, len(cropped_frames), fps):
            # we want every clip to be 1 second
            if len(cropped_frames[i:i+fps]) < fps:
                break

            video_save_path = path.join(video_save_dir, f'{i//fps}'.zfill(fname_pad))
            save_frames = cropped_frames[i:i+fps]

            makedirs(video_save_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
            writer = cv2.VideoWriter(f'{video_save_path}.avi', fourcc, fps, save_frames[0].shape[:2])
            for frame in save_frames:
                writer.write(frame)
            writer.release()

        if path.isdir(video_save_dir):
            with open(path.join(video_save_dir, 'done'), 'w') as _:
                # a dummy file to show that this video has fully procced
                pass

def get_argparser():
    parser = argparse.ArgumentParser(description='Face detector')
    parser.add_argument("--data_dir", required=True, type=str, help="dataset root directory")
    parser.add_argument("--save_dir", required=True, type=str, help="output root directory")
    parser.add_argument('--mean-face', default='20words_mean_face.npy', help='mean face path')
    parser.add_argument('--crop-width', default=250, type=int, help='width of face crop')
    parser.add_argument('--crop-height', default=250, type=int, help='height of face crop')
    parser.add_argument('--start-idx', default=15, type=int, help='start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed landmarks')
    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)
