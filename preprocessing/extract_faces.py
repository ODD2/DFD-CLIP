import os
from os.path import exists
import argparse


import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")

SAMPLE_RATE = 16_000


def load_args():
    parser = argparse.ArgumentParser(description='Pre-processing')
    parser.add_argument('--root-dir', default=None, help='video directory')
    parser.add_argument('--mean-face', default='./preprocessing/20words_mean_face.npy', help='mean face path')
    parser.add_argument('--crop-width', default=150, type=int, help='width of face crop')
    parser.add_argument('--crop-height', default=150, type=int, help='height of face crop')
    parser.add_argument('--start-idx', default=15, type=int, help='start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed landmarks')
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--lm-mode', default="pack", type=str, help='landmark extraction mode.')
    parser.add_argument('--video-ext', default="mp4", type=str, help='video file extension')
    parser.add_argument('--video-folder', default="videos", type=str, help='video folder')
    parser.add_argument('--lm-folder', default="landmarks", type=str, help='landmark folder')
    parser.add_argument('--glob-exp', default="", type=str, help='additional glob expressions.')
    parser.add_argument('--save-folder', default="cropped_faces", type=str,
                        help="folder destination to save the process results.")

    parser.add_argument('--replace', action="store_true", default=False)

    args = parser.parse_args()
    return args


def save_video_lossless(filename, vid, frames_per_second, audio_path=None):
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    writer = cv2.VideoWriter(filename + ".avi", fourcc, frames_per_second, (vid[0].shape[1], vid[0].shape[0]))
    for frame in vid:
        writer.write(frame)
    writer.release()  # close the writer

    if audio_path:
        cmd = f'ffmpeg -y -loglevel warning -i {filename + ".avi"} -i {audio_path} -c copy {filename + "temp.avi"}'
        os.system(cmd)
        os.remove(filename + ".avi")
        os.rename(filename + "temp.avi", filename + ".avi")


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


def cut_patch(img, landmarks, height, width, args, threshold=5):
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

    uy, by = int(round(center_y) - round(height)), int(round(center_y) + round(height))
    lx, rx = int(round(center_x) - round(width)), int(round(center_x) + round(width))
    cutted_img = np.copy(img[uy:by, lx:rx])
    # [2:] remove redundant landmarks from the affine transform procedurea
    cutted_landmarks = np.copy(landmarks) - [lx, uy]
    return cutted_img, cutted_landmarks


def crop_patch(frames, landmarks, reference, args):
    crop_frames = []
    crop_landmarks = []
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
        crop_frame, crop_landmark = cut_patch(
            transformed_frame,
            transformed_landmarks[args.start_idx:args.stop_idx],
            args.crop_height // 2,
            args.crop_width // 2,
            args=args
        )
        crop_frames.append(crop_frame)
        crop_landmarks.append(crop_landmark)
    return np.array(crop_frames), np.array(crop_landmarks)


def get_video_clip(video_filename, stride=1):
    cap = cv2.VideoCapture(video_filename)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    i = -1
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if not i % stride == 0:
            continue

        frames.append(frame.copy())
    cap.release()
    return fps, frames


# def get_video_clip(video_filename, stride=1):
#     vid_reader = VideoReader(
#         video_filename,
#         "video", num_threads=1
#     )

#     fps = round(vid_reader.get_metadata()["video"]["fps"][0])
#     frames = []
#     for i, frame in enumerate(vid_reader):
#         if not i % stride == 0:
#             continue
#         frames.append(cv2.cvtColor(frame["data"].permute((1, 2, 0)).numpy(), cv2.COLOR_BGR2RGB))
#     del vid_reader
#     return fps, frames


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
    args = load_args()
    reference = np.load(args.mean_face)

    video_root = os.path.join(args.root_dir, args.video_folder)
    lm_root = os.path.join(args.root_dir, args.lm_folder)
    target_folder = os.path.join(args.root_dir, args.save_folder)

    for path in tqdm(list(Path(video_root).rglob(f"{args.glob_exp}.{args.video_ext}"))):
        try:
            relpath = os.path.relpath(path, video_root)

            video_path = os.path.join(video_root, relpath)

            target_video_path_wo_ext = os.path.join(target_folder, relpath[:-4])
            target_landmark_path_wo_ext = target_video_path_wo_ext.replace(
                args.save_folder, args.save_folder + "(landmark)"
            )

            if (exists(f"{target_video_path_wo_ext}.avi") and exists(f"{target_landmark_path_wo_ext}.npy") and not args.replace):
                continue

            landmarks = get_video_landmark(video_path, relpath, lm_root, mode=args.lm_mode, stride=args.stride)

            fps, frames = get_video_clip(video_path, stride=args.stride)

            crop_frames, crop_landmarks = crop_patch(frames, landmarks, reference, args)

            # save video
            os.makedirs(os.path.dirname(target_video_path_wo_ext), exist_ok=True)

            save_video_lossless(target_video_path_wo_ext, crop_frames, fps, None)

            # save landmark
            os.makedirs(os.path.dirname(target_landmark_path_wo_ext), exist_ok=True)

            np.save(target_landmark_path_wo_ext + ".npy", crop_landmarks)

        except Exception as e:
            print(f"Error Occur:{path} '{e}'")


if __name__ == "__main__":
    main()
