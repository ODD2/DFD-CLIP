from os import path, scandir, makedirs
import random
import pickle
import json
import cv2
import math

import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from tqdm.auto import tqdm
from yacs.config import CfgNode as CN


import pandas as pd
import heartpy as hp
import xml.etree.ElementTree as ET
from glob import glob
from scipy.signal import resample
from pyedflib import highlevel as reader

import logging


class FFPP(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.name = 'train'
        C.root_dir = './datasets/ffpp/'
        C.types = ['REAL', 'DF']
        C.compressions = ['raw']
        C.detection_level = 'video'
        C.dataset = "FFPP"
        return C

    def __init__(self, config,num_frames,clip_duration, transform, accelerator, split='train'):
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
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.split = split
        self.transform = transform
        if accelerator:
            with accelerator.main_process_first():
                self._build_video_table(accelerator)
        else:
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


class SessionMeta:
    def __init__(self,session_dir,save_gae=False,save_xml=False,cropped_dir=None):
        self.session_dir = session_dir
        self.session_path = path.join(session_dir,"session.xml")
        self.cropped_dir = cropped_dir
        self.xml = None
        self.session_video_beg_sample=None
        self.session_video_end_sample=None
        self.session_video_sample_freq=None
        self.session_audio_beg_sample=None
        self.session_audio_end_sample=None
        self.session_audio_sample_freq=None
        self.session_hr_sample_freq = None
        self.flag_video_beg_sample=None
        self.flag_audio_beg_sample=None
        self.flag_hr_beg_sample=None
        self.video_path = None
        self.bdf_path = None
        self.gae_path = None
        self.gae_data = None
        self.gae_beg_time = None
        self.media_beg_time_ms = None
        self.media_end_time_ms = None
        self.duration = None

        # fetch session xml
        self.xml = ET.parse(self.session_path).getroot()
        # fetch session media setups
        self.session_video_beg_sample = int(float(self.xml.attrib["vidBeginSmp"]))+1
        self.session_video_end_sample = int(float(self.xml.attrib["vidEndSmp"]))
        self.session_video_sample_freq= round(float(self.xml.attrib["vidRate"]))
        self.session_audio_beg_sample = int(float(self.xml.attrib["audBeginSmp"]))+1
        self.session_audio_end_sample = int(float(self.xml.attrib["audEndSmp"]))
        self.session_audio_sample_freq = round(float(self.xml.attrib["audRate"]))

        # fetch paths
        for l1_tag in self.xml:
            if "color" in l1_tag.attrib and  l1_tag.attrib["color"] == "1":
                self.video_path = path.join(
                    self.session_dir if not cropped_dir else self.cropped_dir, l1_tag.attrib["filename"]
                )
                for l2_tag in l1_tag: 
                    if("type" in l2_tag.attrib and l2_tag.attrib["type"] == "Gaze"):
                        self.gae_path = path.join(self.session_dir, l2_tag.attrib["filename"])
            if "type" in l1_tag.attrib and l1_tag.attrib["type"] == "Physiological":
                self.bdf_path = path.join(self.session_dir, l1_tag.attrib["filename"])

        # path check
        file_missing_msg = []
        if(not self.video_path):
            file_missing_msg.append("missing rgb video path")
        if(not self.gae_path):
            file_missing_msg.append("missing gae file path")
        if(not self.bdf_path):
            file_missing_msg.append("missing bdf file path")
        
        if(len(file_missing_msg) > 0):
            logging.info(f"Session: {session_dir}, {', '.join(file_missing_msg)}")

        # prefetch required datas
        self.load_gae_data()
        self.fetch_bdf_infos()
        self.fetch_gae_infos()
        self.sync_time()

        # despose gae to save memory
        if(not save_gae):
            self.gae_data = None
        if(not save_xml):
            self.xml = None
                 
    def load_gae_data(self):
        if(not type(self.gae_data)==type(None)):
            pass
        elif(self.gae_path):
            try:
                self.gae_data =  pd.read_csv(self.gae_path,sep="\t",skiprows=23)
            except Exception as e :
                logging.info(f"unable to load gae data {self.session_dir}, due to '{e}'")

    def fetch_bdf_infos(self):
        if(self.bdf_path):
            signals, signal_headers, header = reader.read_edf(self.bdf_path,ch_names=["EXG1"])
            self.session_hr_sample_freq = signal_headers[0]["sample_frequency"]
            del signals,signal_headers,header
    
    def fetch_gae_infos(self):
        if(not type(self.gae_data)==type(None)):
            media_events = self.gae_data[self.gae_data["Event"].isin(["MovieStart","MovieEnd","ImageStart","ImageEnd"])][["Event","Timestamp"]].to_numpy()
            if(len(media_events) >= 2 and media_events[0][0][-5:] == "Start" and media_events[-1][0][-3:] == "End"):    
                self.gae_beg_time = self.gae_data["Timestamp"].iloc[0]
                self.media_beg_time_ms = media_events[ 0][1]
                self.media_end_time_ms = media_events[-1][1]
                self.duration = (self.media_end_time_ms - self.media_beg_time_ms)//1000

    def sync_time(self):
        if(not type(self.gae_data)==type(None) and self.bdf_path):
            # synchronize the timings of the video, audio, and physio signals.
            gae_anchor_audio_time,gae_anchor_audio_sample = self.gae_data[self.gae_data["AudioSampleNumber"].notnull()][["Timestamp","AudioSampleNumber"]].iloc[0]
            estimate_media_audio_sample = self.audio_sample_shift(self.media_beg_time_ms - gae_anchor_audio_time) + gae_anchor_audio_sample

            if(self.session_audio_beg_sample > estimate_media_audio_sample):
                # the gae records for the media begin time is earlier than the provided video/audio initial timestamp.
                ms_shift = self.audio_time_shift(self.session_audio_beg_sample - estimate_media_audio_sample)
                self.media_beg_time_ms += ms_shift
                self.flag_audio_beg_sample = self.session_audio_beg_sample
                self.flag_hr_beg_sample = int((30 + ms_shift/1000) * self.session_hr_sample_freq)
            else:
                self.flag_audio_beg_sample = estimate_media_audio_sample
                self.flag_hr_beg_sample = int(30 * self.session_hr_sample_freq)
            
            self.flag_video_beg_sample = (self.session_video_beg_sample + 
                int(
                    (self.flag_audio_beg_sample-self.session_audio_beg_sample)/
                    self.session_audio_sample_freq * self.session_video_sample_freq
                )
            )
            self.duration = (self.media_end_time_ms - self.media_beg_time_ms)//1000
            
    def audio_sample_shift(self,ms):
        return ms / 1000 * self.session_audio_sample_freq

    def audio_time_shift(self,sample):
        return sample / self.session_audio_sample_freq * 1000

    def video_length(self):
        return self.session_video_end_sample - self.session_video_beg_sample + 1

    def require_check(self,video=False, gae=False, bdf=False,time=False):
        if(video and not (self.video_path and path.exists(self.video_path))):
            return False
        if(gae and type(self.gae_data)==type(None)):
            return False
        if(bdf and not (self.bdf_path and path.exists(self.bdf_path))):
            return False
        if(time and not self.duration):
            return False

        return True


class RPPG(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.name = 'train'
        C.root_dir = './datasets/hci/'
        C.detection_level = 'video'
        C.train_ratio = 0.95
        C.cropped_folder="cropped_faces"
        C.meta_folder="Metas"
        C.dataset="RPPG"
        return C

    def __init__(self, config,num_frames,clip_duration, transform=None, accelerator=None, split='train'):
        # TODO: accelerator not implemented
        self.name = config.name
        # HCI datasets recorded videos with 61 fps
        self.transform = transform
        self.num_frames = num_frames
        self.clip_duration = clip_duration

        # dataset consistency
        rng = random.Random()
        rng.seed(777)
        session_dirs = sorted(glob(path.join(config.root_dir,"Sessions","*")))
        rng.shuffle(session_dirs)

        # dataset splitting
        if split == "train":
            target_sessions = session_dirs[:int(len(session_dirs)*config.train_ratio)]
        elif split == "val":
            target_sessions = session_dirs[int(len(session_dirs)*config.train_ratio):]
        
        # speed up dataset initialization
        if (not config.meta_folder):
            self.session_metas = [
                SessionMeta(
                    session_dir,
                    cropped_dir=session_dir.replace("Sessions",config.cropped_folder) if config.cropped_folder else None 
                ) 
                for session_dir in target_sessions
            ]
        else:
            self.session_metas = [None for _ in target_sessions]
            for i,session_dir in enumerate(target_sessions):
                try:
                    with open(path.join(session_dir.replace("Sessions",config.meta_folder),"meta.pickle"),"rb") as file:
                        self.session_metas[i] = pickle.load(file)
                except Exception as e:
                    logging.debug(f"Error while loading meta pickle: '{e}'")

        # remove erroneous datas.
        self.session_metas = [meta for meta in self.session_metas if meta and meta.require_check(video=True,bdf=True,time=True)]

        # calculate available clips per session
        self.session_clips = [int(meta.duration // self.clip_duration) for meta in self.session_metas]

        # stacking up the amount of session clips for further usage
        self.stack_session_clips = [0]
        for i in self.session_clips:
            self.stack_session_clips.append(self.stack_session_clips[-1] + i)
        self.stack_session_clips.pop(0)

    def __len__(self):
        return self.stack_session_clips[-1]
    
    def __getitem__(self,idx):
        result = self.get_dict(idx)
        return result["frames"],result["label"],result["mask"]
        
    def get_dict(self,idx):
        while(True):
            try:
                session_idx =  next(i for i,x in enumerate(self.stack_session_clips) if  idx < x)
                session_meta = self.session_metas[session_idx]
                session_offset_duration =  (idx - (0 if session_idx == 0 else self.stack_session_clips[session_idx-1]))*self.clip_duration
                # heart rate data processing
                signals, signal_headers, _ = reader.read_edf(session_meta.bdf_path,ch_names=["EXG1","EXG2","EXG3","Status"])
                _hr_datas = []
                for hr_channel_idx in range(3):
                    try:
                        assert  int(session_meta.session_hr_sample_freq) == int(signal_headers[hr_channel_idx]["sample_frequency"])
                        # - the ERG sample frequency
                        hr_sample_freq =  session_meta.session_hr_sample_freq
                        # - the amount of samples to skip, including the 30s stimulation offset and session clip offset.
                        hr_sample_offset =  session_meta.flag_hr_beg_sample + int(session_offset_duration*hr_sample_freq)
                        # - the amount of samples for the duration of a clip
                        hr_clip_samples = int(hr_sample_freq*self.clip_duration)
                        # - fetch heart rate data of clip duration
                        _hr_data = signals[hr_channel_idx][hr_sample_offset:hr_sample_offset + hr_clip_samples]
                        # - preprocess the ERG data: filter out the noise.
                        _hr_data = hp.filter_signal(_hr_data, cutoff = 0.05, sample_rate = session_meta.session_hr_sample_freq, filtertype='notch')
                        # - scale down the ERG value to 3.4 max.
                        _hr_data = (_hr_data - _hr_data.min())/(_hr_data.max()-_hr_data.min()) * 3.4
                        # - resample the ERG
                        _hr_data = resample(_hr_data, len(_hr_data) * 4)
                        # - process the ERG data: get measurements.
                        _wd, _measures = hp.process(hp.scale_data(_hr_data),session_meta.session_hr_sample_freq * 4)
                        # - nan/error check
                        if(_measures["bpm"] > 180 or _measures["bpm"] < 41):
                            continue

                        for v in _measures.values():
                            # ignore
                            if type(v)==float and math.isnan(v):
                                break
                        else:
                            # - save for comparison.
                            _hr_datas.append((_hr_data,_measures,_wd))
                    except Exception as e:
                        logging.debug(f"Error occur during heart rate analysis for index {idx}:{e}")
                        continue
                
                if(len(_hr_datas) == 0):
                    raise Exception(f"Unable to process the ERG data for index {idx}")
                
                # get the best ERG measurement result with the sdnn
                best_pair = sorted(_hr_datas,key=lambda x : x[1]["sdnn"])[0]
                hr_data,measures,wd = best_pair[0], best_pair[1], best_pair[2]
                bpm = measures["bpm"]
                1/(pow(2*math.pi,0.5))*pow(math.e,(-pow((31-(72-41)),2)/2))
                label = torch.tensor([1/(pow(2*math.pi,0.5))*pow(math.e,(-pow((k-(bpm-41)),2)/2)) for k in range(180)])

                # video frame processing
                frames = []
                cap = cv2.VideoCapture(session_meta.video_path)
                assert int(session_meta.session_video_sample_freq) == int(cap.get(cv2.CAP_PROP_FPS)), f"{int(session_meta.session_video_sample_freq)},{int(cap.get(cv2.CAP_PROP_FPS))}"
                video_sample_freq = session_meta.session_video_sample_freq
                # - the amount of frames to skip
                video_sample_offset = int(session_meta.flag_video_beg_sample - session_meta.session_video_beg_sample) + int(video_sample_freq * session_offset_duration)
                # - the amount of frames for the duration of a clip
                video_clip_samples = int(session_meta.session_video_sample_freq * self.clip_duration)
                # - the amount of frames to skip in order to meet the num_frames per clip.(excluding the head & tail frames )
                video_sample_stride = video_clip_samples / (self.num_frames - 1)
                # - fast forward to the the sampling start.
                cap.set(cv2.CAP_PROP_POS_FRAMES,video_sample_offset)
                # - fetch frames of clip duration
                next_sample_idx = 0
                for sample_idx in range(video_clip_samples):
                    ret, frame = cap.read()
                    if(ret):
                        if(sample_idx == next_sample_idx):
                            frames.append(torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).transpose((2,0,1))))
                            next_sample_idx = int(round(len(frames) * video_sample_stride))
                    else:
                        raise NotImplementedError()
                frames = torch.stack(frames)

                # transformation
                if (self.transform):
                    frames = self.transform(frames)

                # padding and masking missing frames.
                mask = torch.tensor([1.] * len(frames) +
                                    [0.] * (self.num_frames - len(frames)), dtype=torch.bool)
                if len(frames) < self.num_frames:
                    diff = self.num_frames - len(frames)
                    padding = torch.zeros((diff, *frames.shape[1:]),dtype=torch.uint8)
                    frames = torch.concatenate((frames, padding))
                
                return {
                    "frames":frames,
                    "label":label,
                    "mask":mask,
                    "hr_data":hr_data,
                    "measures":measures,
                    "wd":wd
                }
            except Exception as e:
                logging.error(f"Error occur: {e}")
                idx = random.randrange(0,len(self))

    def save_meta(self,meta_folder="Metas"):
        for meta in self.session_metas:
            meta_dir = meta.session_dir.replace("Sessions",meta_folder)
            makedirs(meta_dir,exist_ok=True)
            with open(path.join(meta_dir,"meta.pickle"),"wb") as file:
                pickle.dump(obj=meta,file=file)
