import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import face_alignment


def landmark_extract(fn, output_pipe, input_pipe, org_path, save_path, stride, batch_size, frame_source=False):

    cap_org = cv2.VideoCapture(
        org_path if not frame_source else os.path.join(org_path, "00000.jpg")
    )
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    land_path = save_path

    video_landmark = []
    video_bbox = []

    batch_frames = []
    frame_indexes = []

    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        height, width = frame_org.shape[:-1]
        if not ret_org:
            raise Exception('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(org_path)))

        if cnt_frame % stride == 0:
            frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
            frame_indexes.append(cnt_frame)

        if (len(batch_frames) == batch_size or (cnt_frame == (frame_count_org - 1) and len(batch_frames) > 0)):
            with torch.inference_mode():
                batch_faces = output_pipe(fn(input_pipe(batch_frames)))
            for faces in batch_faces:
                try:
                    if len(faces) == 0:
                        raise Exception('No faces in {}:{}'.format(cnt_frame, os.path.basename(org_path)))

                    face_s_max = -1
                    landmarks = None
                    bbox = None
                    for face_idx in range(len(faces)):
                        x0, y0, x1, y1 = faces[face_idx]['bbox']
                        _landmark = faces[face_idx]['landmarks']
                        _bbox = faces[face_idx]['bbox']
                        face_s = (x1 - x0) * (y1 - y0)
                        if (face_s > face_s_max):
                            face_s_max = face_s
                            landmarks = _landmark
                            bbox = _bbox
                except Exception as e:
                    raise e
                video_landmark.append(landmarks)
                video_bbox.append(bbox)
            batch_frames.clear()
            frame_indexes.clear()

    np.save(land_path, np.stack(video_landmark, 0))
    cap_org.release()


def fan_input_pipe(batch):
    return torch.tensor(np.stack(batch).transpose((0, 3, 1, 2)))


def fan_output_pipe(datas):
    batch_size = len(datas[0])
    faces_datas = []
    for i in range(batch_size):
        faces = []
        landmarks = datas[0][i]
        for j, bbox in enumerate(datas[2][i]):
            faces.append(
                {
                    "landmarks": landmarks[68 * j:68 * (j + 1)],
                    "bbox": bbox[:-1]
                }
            )
        faces_datas.append(faces)
    return faces_datas


def extract_landmarks(file_path):
    model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda")
    def fan_extractor(x): return model.get_landmarks_from_batch(x, return_bboxes=True)

    _, video_ext = os.path.splitext(file_path)

    save_path = file_path.replace(video_ext, '.npy')

    landmark_extract(fan_extractor, fan_output_pipe, fan_input_pipe, file_path, save_path, 1, 8, frame_source=False)

##############################################################################################################################


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


def crop_patch(frames, landmarks, reference, window_margin=12, crop_size=150, start_idx=15, stop_idx=68):
    sequence = []
    length = min(len(landmarks), len(frames))
    for frame_idx in range(length):
        frame = frames[frame_idx]
        window_margin = min(window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
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
                transformed_landmarks[start_idx:stop_idx],
                crop_size // 2,
                crop_size // 2,
            )
        )
    return np.array(sequence)


def get_video_clip(video_path, stride=1):
    cap = cv2.VideoCapture(video_path)
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


def get_video_landmark(landmark_path, video_path, stride=1):

    landmarks = []
    cap = cv2.VideoCapture(video_path)
    frame_count_org = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count_org > 0
    cap.release()

    landmarks_pack = np.load(landmark_path)
    landmarks = [lm for i, lm in enumerate(landmarks_pack) if i % stride == 0]

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


def get_cropped_videos(video_path):
    reference = np.load("misc/20words_mean_face.npy")
    _, video_ext = os.path.splitext(video_path)
    landmark_path = video_path.replace(video_ext, ".npy")
    video_folder, video_name = os.path.split(video_path)
    crop_video_name = os.path.join(video_folder, f"cropped_{video_name[:-len(video_ext)]}")

    landmarks = get_video_landmark(landmark_path, video_path, stride=1)

    fps, frames = get_video_clip(video_path, stride=1)

    sequence = crop_patch(frames, landmarks, reference)

    save_video_lossless(crop_video_name, sequence, fps)

##############################################################################################################################


import torchvision
import pickle
import json

from accelerate import Accelerator
import evaluate
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN
from tqdm import tqdm
import yaml
from src.models import Detector


def get_config(cfg_file):
    with open(cfg_file) as f:
        preset = CN(yaml.safe_load(f))

    C = CN()

    # prerequisite: fetch the Deepfake detection task index during training.
    C.target_task = next(i for i, d in enumerate(preset.data.eval) if d.category == "Deepfake")

    C.data = CN()
    C.data.num_frames = preset.data.num_frames
    C.data.clip_duration = preset.data.clip_duration

    # model
    C.model = Detector.get_default_config().merge_from_other_cfg(preset.model)

    C.freeze()
    return C


def collate_fn(batch):
    # just forward the dataset
    # only work with batch_size=1
    return batch[0]


@torch.no_grad()
def get_result(video_path, weight_path, cfg_name="setting"):

    _, video_ext = os.path.splitext(video_path)
    video_folder, video_name = os.path.split(video_path)
    video_path = os.path.join(video_folder, f"cropped_{video_name[:-len(video_ext)]}.avi")

    N = 16
    config = get_config(os.path.join(weight_path, f"{cfg_name}.yaml"))

    stride = config.data.clip_duration / config.data.num_frames

    accelerator = Accelerator()

    model = Detector(config.model, config.data.num_frames, accelerator).to(accelerator.device).eval()

    transform = model.transform

    with accelerator.main_process_first():
        model.load_state_dict(torch.load(os.path.join(weight_path, f'best_weights.pt')))

    vid_reader = torchvision.io.VideoReader(
        video_path,
        "video"
    )
    # - frames per second
    vid_duration = vid_reader.get_metadata()["video"]["duration"][0]
    frames = []
    mask = []
    # - fetch frames of clip duration
    for t in np.arange(0, vid_duration, stride):
        try:
            vid_reader.seek(t)
            frame = next(vid_reader)
            frames.append(transform(frame["data"]))
            mask.append(torch.tensor(1, dtype=bool))
        except Exception as e:
            raise Exception(f"unable to read video frame of sample index:{t}")
    del vid_reader

    clips = []
    masks = []
    for i in range(0, len(frames), config.data.num_frames):
        clips.append(torch.stack(frames[i:i + config.data.num_frames]))
        masks.append(torch.stack(mask[i:i + config.data.num_frames]))

    if (len(clips[-1]) < config.data.num_frames):
        clips = clips[:-1]
        masks = masks[:-1]

    logits = []
    for i in range(0, len(clips), N):
        logits.append(
            model.predict(
                torch.stack(clips[i:i + N]).to(accelerator.device),
                torch.stack(masks[i:i + N]).to(accelerator.device)
            )[0][0].detach().to("cpu")
        )

    p = torch.cat(logits).softmax(dim=-1)

    pred_prob = p.mean(dim=0).unsqueeze(0)

    return pred_prob[0][1]


if __name__ == "__main__":
    extract_landmarks("/home/od/Desktop/test/195_442.avi")
    get_cropped_videos("/home/od/Desktop/test/195_442.avi")
    print(get_result(
        "/home/od/Desktop/test/195_442.avi",
        "/home/od/Desktop/repos/dfd-clip/logs/deepfake/deepfake/c23+resi+dbal+3e-3+4s20f+reg_vien(hf,s,pfa-s)+last6+SGD(0.9m)",
        "config"
    ))
