import os
import cv2
import torch
import argparse
import torchvision
import numpy as np
from math import ceil
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from preprocessing.extract_faces import get_video_clip, save_video_lossless

import cv2
import facer
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from preprocessing.extract_faces import get_video_clip, save_video_lossless

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.inference_mode()
def get_semantics(frames, landmarks, face_parser, batch_size=50):
    # stack frames
    frames = np.stack(frames)
    frames = frames.transpose((0, 3, 1, 2))
    frames = torch.from_numpy(frames)
    # preprocess landmarks to match falr face parser's requirements.
    landmarks = torch.from_numpy(
        np.stack([
            np.stack([
                np.mean(landmarks[f, idxs - 16], axis=0) for idxs in [
                    np.array([i for i in range(37, 43)]),
                    np.array([i for i in range(43, 49)]),
                    np.array([34]),
                    np.array([49]),
                    np.array([55])
                ]
            ]) for f in range(landmarks.shape[0])
        ])
    ).float()

    # facial semantic parsing

    seg_logits = []
    for i in range(frames.shape[0] // batch_size + 1):
        batch_frames = frames[i * batch_size:(i + 1) * batch_size]
        batch_landmarks = landmarks[i * batch_size:(i + 1) * batch_size]
        assert batch_frames.shape[0] == batch_landmarks.shape[0]

        if (batch_frames.shape[0] == 0):
            continue
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            faces = face_parser(
                batch_frames.to(device),
                {
                    "points": batch_landmarks.to(device),
                    "image_ids": torch.arange(0, batch_landmarks.shape[0]).to(device)
                }
            )
        seg_logits.append(
            faces["seg"]["logits"].cpu()
        )
    # concat segmentation logits of all frames
    seg_logits = torch.cat(seg_logits, dim=0)

    # post-processing
    n_classes = seg_logits.size(1)
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    seg_label = seg_probs.argmax(dim=1)
    seg_visual = seg_label.float() / n_classes * 255
    seg_label = seg_label.unsqueeze(-1).numpy().astype(np.uint8)
    seg_visual = seg_visual.unsqueeze(-1).numpy().astype(np.uint8)
    seg_label = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in seg_label]
    seg_visual = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in seg_visual]
    return (
        seg_label,
        seg_visual
    )


def load_args():
    parser = argparse.ArgumentParser(description='Pre-processing')
    parser.add_argument('--root-dir', default=None, help='video directory')
    parser.add_argument('--lm-mode', default="pack", type=str, help='landmark extraction mode.')
    parser.add_argument('--video-ext', default="avi", type=str, help='video file extension')
    parser.add_argument('--video-folder', default="cropped_faces", type=str, help='video folder')
    parser.add_argument('--glob-exp', default="*/*", type=str, help='additional glob expressions.')
    parser.add_argument('--save-folder', default="semantic_videos", type=str,
                        help="folder destination to save the process results.")
    parser.add_argument('--replace', action="store_true", default=False)
    parser.add_argument('--stride', default=1, type=int)

    args = parser.parse_args()
    return args


def main():
    args = load_args()

    face_parser = facer.face_parser(
        'farl/lapa/448', device=device
    )

    video_root = os.path.join(args.root_dir, args.video_folder)
    target_root = os.path.join(args.root_dir, args.save_folder)

    for path in tqdm(list(Path(video_root).rglob(f"{args.glob_exp}.{args.video_ext}"))):
        try:
            relpath = os.path.relpath(path, video_root)

            video_path = os.path.join(video_root, relpath)

            landmark_path = video_path.replace(
                args.video_folder, args.video_folder + "(landmark)"
            ).replace(args.video_ext, "npy")

            target_path_wo_ext = os.path.join(target_root, relpath[:-4])

            if (os.path.exists(f"{target_path_wo_ext}.avi") and not args.replace):
                continue

            fps, frames = get_video_clip(video_path, stride=args.stride)

            landmarks = np.load(landmark_path)

            semantic_frames, notation_frames = get_semantics(frames, landmarks, face_parser)

            os.makedirs(os.path.dirname(target_path_wo_ext), exist_ok=True)

            save_video_lossless(target_path_wo_ext, semantic_frames, fps, None)
            save_video_lossless(f"{target_path_wo_ext}_notate", notation_frames, fps, None)

        except Exception as e:
            print(f"Error Occur:{path} '{e}'")


if __name__ == "__main__":
    main()
