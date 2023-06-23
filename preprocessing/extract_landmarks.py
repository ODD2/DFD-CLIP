import os
import cv2
import math
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import face_alignment

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--lm-dir", type=str, default="landmarks")
    parser.add_argument("--ff-dir", type=str, default="")
    parser.add_argument("--glob-exp", type=str, default="*")
    parser.add_argument("--frame-source", action="store_true", default=False)
    parser.add_argument("--split-num", type=int, default=1)
    parser.add_argument("--part-num", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--save-mode", type=str, default="pack")
    parser.add_argument("--skip-face",action="store_true",default=False)
    args =  parser.parse_args()
    assert args.part_num > 0 and args.split_num > 0, "split and part value should be > 0"
    
    args.part_num = args.part_num -1
    
    return args

class FaceData:
    def __init__(self, _lm, _bbox, _idx):
        self.ema_lm = _lm
        self.lm = [_lm]
        self.bbox = [_bbox]
        self.idx = [_idx]

    def last(self):
        return self.ema_lm

    def add(self, _lm, _bbox, _idx):
        self.ema_lm = self.ema_lm*0.5 +_lm * 0.5
        self.lm.append(_lm)
        self.bbox.append(_bbox)
        self.idx.append(_idx)

    def __len__(self):
        return len(self.lm)


def landmark_extract(fn, org_path, stride, batch_size, frame_source=False, require_face=True):

    cap_org = cv2.VideoCapture(
        org_path if not frame_source else os.path.join(org_path, "00000.jpg")
    )
    try:
        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_faces = [None for _ in range(frame_count_org)]
        batch_indices = []
        batch_frames = []

        scale = None

        for cnt_frame in range(frame_count_org):
            ret_org, frame_org = cap_org.read()
            height, width = frame_org.shape[:-1]

            if scale == None:
                # determine the scaling factor to shrink the size of input image(for efficiency).
                if (max(height, width) > 800):
                    scale = 800 / max(height, width)
                else:
                    scale = 1

            if not ret_org:
                tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(org_path)))
                continue

            if cnt_frame % stride == 0:
                frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
                batch_frames.append(frame)
                batch_indices.append(cnt_frame)

            if (len(batch_frames) == batch_size or (cnt_frame == (frame_count_org - 1) and len(batch_frames) > 0)):
                with torch.inference_mode():
                    batch_frames = [cv2.resize(frame, None, fx=scale, fy=scale) for frame in batch_frames]
                    results = fn(torch.tensor(np.stack(batch_frames).transpose((0, 3, 1, 2))))
                    batch_size = len(results[0])
                    batch_faces = []
                    for i in range(batch_size):
                        faces = []
                        landmarks = results[0][i]
                        for j, bbox in enumerate(results[2][i]):
                            faces.append(
                                {
                                    "landmarks": landmarks[68 * j:68 * (j + 1)] / scale,
                                    "bbox": bbox[:-1] / scale
                                }
                            )
                        batch_faces.append(faces)

                if (require_face):
                    for faces in batch_faces:
                        if (len(faces) == 0):
                            raise Exception(f"require_face == true, but no faces detected in frame of {org_path}.")
                        
                for index, faces in zip(batch_indices,batch_faces):
                    frame_faces[index] = faces

                batch_frames.clear()
                batch_indices.clear()



        # post-process the extracted frame faces.
        # create face identity database to track landmark motion.
        face_dbs = []
        for  index, faces in enumerate(frame_faces):

            if(faces == None):
                # the frame was ignored, might be the stride setting.
                continue

            try:
                # the frame was processed, but no faces can be detected.
                if len(faces) == 0:
                    print(faces)
                    tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(org_path)))
                    continue

                frame_landmarks = np.stack([face["landmarks"] for face in faces])
                matched_indices = []

                for face_data in face_dbs:

                    lm_diff = np.sum(
                        np.linalg.norm(frame_landmarks - face_data.last(), axis=-1),
                        axis=1
                    ) / frame_landmarks.shape[1]

                    # the motion continues if the landmark motion distance is lower than 100.
                    if (np.min(lm_diff) > 100):
                        continue

                    closest_idx = np.argmin(lm_diff)
                    matched_indices.append(closest_idx)
                    face_data.add(faces[closest_idx]["landmarks"], faces[closest_idx]["bbox"], index)

                # create new database entity for untracked landmarks.
                for i, face in enumerate(faces):
                    if i in matched_indices:
                        continue
                    else:
                        _landmark = face['landmarks']
                        _bbox = face['bbox']
                        face_dbs.append(FaceData(_landmark, _bbox, index))
            except Exception as e:
                print(f'error in {cnt_frame}:{org_path}')
                print(e)
                continue
        
        # report only the most consistant face in the video.
        dominant_face = sorted(face_dbs, key=lambda x: len(x), reverse=True)[0]

        return frame_count_org, dominant_face.lm, dominant_face.bbox, dominant_face.idx, frame_faces
    except Exception as e:
        raise e
    finally:
        cap_org.release()

def main():
    # This file extract video landmarks from a given folder.
    # In addition, the landmarks are tracked with landmarks from previous frames.
    # By doing so, we expect to extract the most consistently appeared faces from a given video.
    # Note that under 'pack' save mode, the extracted faces must match the length of the video.
    # That's to say, if there exists a single frame without appearing faces in the video, the extract operation fails.

    args = load_args()

    model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector='sfd',
        dtype=torch.float16, # float16 to boost efficiency.
        flip_input=False,
        device="cuda",
    )

    def driver(x): return model.get_landmarks_from_batch(x, return_bboxes=True)

    if (not args.root_dir[-1] == "/"):
        args.root_dir += "/"

    video_files = sorted(glob(os.path.join( args.root_dir,args.video_dir, args.glob_exp)))
    _, video_ext = os.path.splitext(video_files[0])

    # splitting
    split_size = math.ceil(len(video_files) / args.split_num)
    video_files = video_files[args.part_num * split_size:(args.part_num + 1) * split_size]
    n_videos = len(video_files)

    print("{} videos in {}".format(n_videos, args.root_dir))
    print("path sample:{}".format(video_files[0]))

    cont = input(f"Processing Part {args.part_num+1}/{args.split_num}, Confirm?(y/n)")
    if (not cont.lower() == "y"):
        print("abort.")
        return

    for i in tqdm(range(n_videos)):
        try:
            if not args.ff_dir == "":
                face_path = video_files[i].replace(args.video_dir, args.ff_dir).replace(video_ext, '.pickle')
            else:
                face_path = None

            if "pack" == args.save_mode:
                lm_path = video_files[i].replace(args.video_dir, args.lm_dir).replace(video_ext, '.npy')
            elif "frame"  == args.save_mode:
                lm_path = video_files[i].replace(args.video_dir, args.lm_dir).replace(video_ext, '') + "/"
            
            if (os.path.exists(lm_path) and (face_path == None or os.path.exists(face_path))):
                continue

            total_frames, lm, bbox, idx, frame_faces = landmark_extract(
                driver, video_files[i],
                args.stride, args.batch, frame_source=args.frame_source,
                require_face=not args.skip_face
            )

            if not face_path == None:
                os.makedirs(os.path.split(face_path)[0],exist_ok=True)
                with open(face_path,"wb") as f:
                    pickle.dump(frame_faces, f)

            if not args.skip_face:
                if not total_frames == len(lm):
                    raise Exception(f"{video_files[i]} cannot fully fetch frame landmarks for pack mode, ignoring video")
                else:
                    assert len(lm) == len(bbox) == len(idx), "landmarks, bboxes, and frame indices should be the same size."

            if "pack" ==  args.save_mode:
                os.makedirs(os.path.split(lm_path)[0],exist_ok=True)
                np.save(lm_path, lm)

            elif "frame" == args.save_mode:
                for landmark, bbox, index in zip(lm, bbox, idx):
                    land_path = os.path.join(lm_path, str(index).zfill(3) + ".npy")
                    os.makedirs(os.path.dirname(land_path), exist_ok=True)
                    np.save(land_path, landmark)
            else:
                raise NotImplementedError()
            

        except NotImplementedError as e:
            raise e
        except Exception as e:
            print(f"Error Occur: {e}")
        
if __name__ == '__main__':
    main()
