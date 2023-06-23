import os
from os.path import exists
import argparse
import pickle


import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")

def load_args():
    parser = argparse.ArgumentParser(description='Pre-processing')
    parser.add_argument('--root-dir', default=None, help='video directory')
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--lm-mode', default="pack", type=str, help='landmark extraction mode.')
    parser.add_argument('--video-ext', default="mp4", type=str, help='video file extension')
    parser.add_argument('--video-folder', default="videos", type=str, help='video folder')
    parser.add_argument('--lm-folder', default="landmarks", type=str, help='landmark folder')
    parser.add_argument('--glob-exp', default="", type=str, help='additional glob expressions.')
    args = parser.parse_args()
    return args


def get_video_landmark(video_path, video_relpath, landmarks_root, mode="pack", stride=1):
    assert mode in ["pack", "frame"]
    landmarks = []
    landmarks_base_path = os.path.join(landmarks_root, video_relpath[:-4])
    cap = cv2.VideoCapture(video_path)
    frame_count_org = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count_org > 0
    cap.release()

    if mode == "pack":
        landmarks_pack = np.load(landmarks_base_path + ".npy")
        landmarks = [lm for i, lm in enumerate(landmarks_pack) if i % stride == 0]
    elif mode == "frame":
        for cnt_frame in range(frame_count_org):
            if (not cnt_frame % stride == 0):
                continue
            lm_path = os.path.join(landmarks_base_path, str(cnt_frame).zfill(3) + ".npy")
            landmarks.append(np.load(lm_path))
    if (len(landmarks[0]) == 98):
        _98_to_68_mapping = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
            26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44,
            45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76,
            77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
            89, 90, 91, 92, 93, 94, 95
        ]

        landmarks = [lm[_98_to_68_mapping]for lm in landmarks]
    assert len(landmarks[0]) == 68
    return landmarks


def main():
    "This file examinates the extracted video landmarks are consistent thoroughly."
    args = load_args()

    video_root = os.path.join(args.root_dir, args.video_folder)
    lm_root = os.path.join(args.root_dir, args.lm_folder)
    files = []
    for path in tqdm(list(Path(video_root).rglob(f"{args.glob_exp}.{args.video_ext}"))):
        try:
            relpath = os.path.relpath(path, video_root)

            video_path = os.path.join(video_root, relpath)

            landmarks = get_video_landmark(video_path, relpath, lm_root, mode=args.lm_mode, stride=args.stride)

            # video landmarks
            landmarks = np.array(landmarks)

            # calculate landmark shift across neighboring two frames.
            ldiff = landmarks[1:] - landmarks[:-1]
            # calculate norm of the shift
            ldist = np.linalg.norm(ldiff,axis=-1)
            # quantify the landmark shift across neighboring two frames
            ldist = np.sum(ldist, axis=1) / landmarks.shape[1]
            # validate the distance of the maximum and mean shift values are in the desire threshold.
            # if not, the extracted landmarks are highly possible from multiple faces in the video clip, 
            # rather than a single one, which is what we desire.
            if ((np.max(ldist) - np.mean(ldist)) > 100):
                files.append(video_path)

        except Exception as e:
            print(f"Error Occur:{path} '{e}'")

    with open("./out.pickle", "wb") as f:
        pickle.dump(files, f)


if __name__ == "__main__":
    main()
