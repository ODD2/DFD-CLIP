import os
import cv2
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from retinaface.pre_trained_models import get_model


def landmark_extract(fn, output_pipe, input_pipe, org_path, save_path, stride, batch_size, frame_source=False):

    cap_org = cv2.VideoCapture(
        org_path if not frame_source else os.path.join(org_path, "00000.jpg")
    )
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    batch_frames = []
    frame_indexes = []
    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        height, width = frame_org.shape[:-1]
        if not ret_org:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(org_path)))
            continue

        land_path = save_path + str(cnt_frame).zfill(3) + ".npy"
        bbox_path = land_path.replace("landmark", 'retina')
        if cnt_frame % stride == 0:
            if os.path.exists(land_path):
                continue

            frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
            frame_indexes.append(cnt_frame)

        if(len(batch_frames) == batch_size or (cnt_frame == (frame_count_org - 1) and len(batch_frames) > 0)):
            with torch.inference_mode():
                batch_faces = output_pipe(fn(input_pipe(batch_frames)))
            for batch_index, faces in enumerate(batch_faces):
                land_path = save_path + str(frame_indexes[batch_index]).zfill(3) + ".npy"
                try:
                    if len(faces) == 0:
                        print(faces)
                        tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(org_path)))
                        continue
                    face_s_max = -1
                    landmarks = None
                    bbox = None
                    for face_idx in range(len(faces)):
                        x0, y0, x1, y1 = faces[face_idx]['bbox']
                        _landmark = faces[face_idx]['landmarks']
                        _bbox = faces[face_idx]['bbox']
                        face_s = (x1 - x0) * (y1 - y0)
                        if(face_s > face_s_max):
                            face_s_max = face_s
                            landmarks = _landmark
                            bbox = _bbox
                except Exception as e:
                    print(f'error in {cnt_frame}:{org_path}')
                    print(e)
                    continue
                os.makedirs(os.path.dirname(land_path), exist_ok=True)
                os.makedirs(os.path.dirname(bbox_path), exist_ok=True)
                np.save(land_path, landmarks)
                np.save(bbox_path, bbox)
            batch_frames.clear()
            frame_indexes.clear()

    cap_org.release()


def retina_input_pipe(batch: torch.Tensor):
    assert len(batch) == 1
    return batch[0]


def retina_output_pipe(data):
    return [data]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest="video_dir", type=str, default="")
    parser.add_argument('-t', dest="save_folder", type=str, default="landmark")
    parser.add_argument('-e', dest="expression", type=str, default="*")
    parser.add_argument('-f', dest="frame_source", action="store_true", default=False)
    parser.add_argument('-s', dest="split_num", type=int, default=1)
    parser.add_argument('-p', dest="part_num", type=int, default=0)
    parser.add_argument('-z', dest="stride", type=int, default=1)
    parser.add_argument('-b', dest="max_batch_size", type=int, default=1)
    parser.add_argument('-m', dest="model", type=str, default="retina")

    args = parser.parse_args()
    device = torch.device('cuda')
    if(args.model == "retina"):

        model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
        model.eval()
        fn = model.predict_jsons
        output_pipe = retina_output_pipe
        input_pipe = retina_input_pipe
    elif(args.model == "fan"):
        import face_alignment
        model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda")
        def driver(x): return model.get_landmarks_from_batch(x, return_bboxes=True)
        fn = driver
        output_pipe = fan_output_pipe
        input_pipe = fan_input_pipe

    if(not args.video_dir[-1] == "/"):
        args.video_dir += "/"

    video_file_list = sorted(glob(os.path.join(args.video_dir, args.expression)))
    video_folder = os.path.split(args.video_dir)[-2].split('/')[-1]
    _, video_ext = os.path.splitext(video_file_list[0])
    # splitting
    split_size = math.ceil(len(video_file_list) / args.split_num)
    video_file_list = video_file_list[args.part_num * split_size:(args.part_num + 1) * split_size]
    n_videos = len(video_file_list)

    print("{} videos in {}".format(n_videos, args.video_dir))
    print("path sample:{}".format(video_file_list[0]))

    cont = input(f"Processing Part {args.part_num+1}/{args.split_num}, Confirm?(y/n)")
    if(not cont.lower() == "y"):
        print("abort.")
        return

    for i in tqdm(range(n_videos)):
        folder_path = video_file_list[i].replace(video_folder, args.save_folder).replace(video_ext, '') + "/"
        landmark_extract(
            fn, output_pipe, input_pipe, video_file_list[i], folder_path,
            args.stride, args.batch_size, frame_source=args.frame_source
        )


if __name__ == '__main__':
    main()
