from os import path, scandir, makedirs
import random
import pickle
import json

import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from tqdm.auto import tqdm
from yacs.config import CfgNode as CN

class FFPP(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.name = 'train'
        C.root_dir = '/home/ernestchu/scratch4/Datasets/FFPP'
        C.types = ['REAL', 'DF']
        C.compressions = ['raw']
        C.detection_level = 'video'
        C.num_frames = 30
        return C

    def __init__(self, config, transform, accelerator, split='train'):
        self.TYPE_DIRS = {
            'REAL': 'data/original_sequences/youtube/',
            # 'DFD' : 'data/original_sequences/actors/',
            'DF'  : 'data/manipulated_sequences/Deepfakes/',
            'FS'  : 'data/manipulated_sequences/FaceSwap/',
            'F2F' : 'data/manipulated_sequences/Face2Face/',
            'NT'  : 'data/manipulated_sequences/NeuralTextures/',
            'FSH' : 'data/manipulated_sequences/FaceShifter/',
            # 'DFD-FAKE' : 'data/manipulated_sequences/DeepFakeDetection/',
        }
        self.name = config.name
        self.root = path.expanduser(config.root_dir)
        self.detection_level = config.detection_level
        self.types = config.types
        self.compressions = config.compressions
        self.num_frames = config.num_frames
        self.split = split
        self.transform = transform
        with accelerator.main_process_first():
            self._build_video_table(accelerator)
        self._build_data_list(accelerator)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.split == 'test':
            # if at testing, return all 1-sec clips of a video
            # TODO: frame-level testing
            return self._get_test_item(idx)

        df_type, comp, idx = self.data_list[idx]
        video_clips = self.video_table[df_type][comp][idx]
        clip  = random.choice(video_clips)
        reader = VideoReader(path.join(self.root, self.TYPE_DIRS[df_type], comp, 'videos', idx, f'{clip}.avi'), "video")
        metadata = reader.get_metadata()
        if self.detection_level == 'frame':
            frame = None
            n_tries = 0
            while frame is None:
                n_tries += 1
                reader.seek(random.uniform(0, metadata['video']['duration'][0]) if n_tries < 3 else 0)
                try:
                    frame = next(reader)['data']
                except StopIteration:
                    pass

            # apply transform
            frame = self.transform(frame)

            return frame, 1 if df_type == 'REAL' else 0

        elif self.detection_level == 'video':
            # TODO: random select start time frame
            frames = []
            count = 0
            for frame in reader:
                count += 1
                if count > self.num_frames:
                    break
                frames.append(frame['data'])

            frames = self.transform(torch.stack(frames))
            mask = torch.tensor([1.] * len(frames) +
                                [0.] * (self.num_frames - len(frames)), dtype=torch.bool)

            if len(frames) < self.num_frames:
                diff = self.num_frames - len(frames)
                padding = torch.zeros((diff, *frames.shape[1:]))
                frames = torch.concatenate((frames, padding))

            return frames, 1 if df_type == 'REAL' else 0, mask
        else:
            raise NotImplementedError

    def _build_video_table(self, accelerator):
        self.video_table = {}

        progress_bar = tqdm(self.types, disable=not accelerator.is_local_main_process)
        for df_type in progress_bar:
            self.video_table[df_type] = {}
            for comp in self.compressions:
                video_cache = path.expanduser(f'~/.cache/dfd-clip/videos/{df_type}-{comp}.pkl')
                if path.isfile(video_cache):
                    with open(video_cache, 'rb') as f:
                        videos = pickle.load(f)
                    self.video_table[df_type][comp] = videos
                    continue

                subdir = path.join(self.root, self.TYPE_DIRS[df_type], '')
                # video table
                videos = {f.name: [] for f in scandir(path.join(subdir, f'{comp}/videos')) if f.is_dir()}
                for fname in videos:
                    # clip list
                    videos[fname] = [
                        f.name[:-4] for f in scandir(path.join(subdir, f'{comp}/videos', fname)) if '.avi' in f.name]
                    progress_bar.set_description(f"{df_type}: {path.join(f'{comp}/videos', fname)}")

                if accelerator.is_local_main_process:
                    makedirs(path.dirname(video_cache), exist_ok=True)
                    with open(video_cache, 'wb') as f:
                        pickle.dump(videos, f)

                self.video_table[df_type][comp] = videos
        
    def _build_data_list(self, accelerator):
        self.data_list = []
        
        with open(path.join(self.root, 'splits', f'{self.split}.json')) as f:
            idxs = json.load(f)
            
        for df_type in self.types:
            for comp in self.compressions:
                adj_idxs = [i for inner in idxs for i in inner] if df_type == 'REAL' else ['_'.join(idx) for idx in idxs] + ['_'.join(reversed(idx)) for idx in idxs]

                for idx in adj_idxs:
                    if idx in self.video_table[df_type][comp]:
                        self.data_list.append((df_type, comp, idx))
                    else:
                        accelerator.print(f'Warning: video {path.join(self.root, self.TYPE_DIRS[df_type], comp, "videos", idx)} does not present in the processed dataset.')

    def _get_test_item(self, idx):
        df_type, comp, idx = self.data_list[idx]
        video_clips = self.video_table[df_type][comp][idx]
        if self.detection_level == 'frame':
            raise NotImplementedError
        elif self.detection_level == 'video':
            clips = []
            masks = []
            for clip in video_clips:
                reader = VideoReader(path.join(self.root, self.TYPE_DIRS[df_type], comp, 'videos', idx, f'{clip}.avi'), "video")
                frames = []
                count = 0
                for frame in reader:
                    count += 1
                    if count > self.num_frames:
                        break
                    frames.append(frame['data'])

                frames = self.transform(torch.stack(frames))
                mask = torch.tensor([1.] * len(frames) +
                                    [0.] * (self.num_frames - len(frames)), dtype=torch.bool)

                if len(frames) < self.num_frames:
                    diff = self.num_frames - len(frames)
                    padding = torch.zeros((diff, *frames.shape[1:]))
                    frames = torch.concatenate((frames, padding))

                clips.append(frames)
                masks.append(mask)

            return clips, 1 if df_type == 'REAL' else 0, masks
        else:
            raise NotImplementedError
