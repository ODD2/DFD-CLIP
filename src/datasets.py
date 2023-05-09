import sys
import os
import random
import pickle
import json
import cv2
import math
from time import time
from os import path, scandir, makedirs

import torch
import torchvision
from torch.utils.data import Dataset,default_collate
from torchvision.io import VideoReader
from tqdm.auto import tqdm
from yacs.config import CfgNode as CN

import traceback

import pandas as pd
import heartpy as hp
import xml.etree.ElementTree as ET
from glob import glob
from scipy.signal import resample
from pyedflib import highlevel as BDFReader
import logging

class SessionMeta:
    def __init__(self,session_dir,save_gae=False,save_xml=False):
        self.session_dir = session_dir
        self.session_path = path.join(session_dir,"session.xml")
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
                    self.session_dir, l1_tag.attrib["filename"]
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
            signals, signal_headers, header = BDFReader.read_edf(self.bdf_path,ch_names=["EXG1"])
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

    def require_check(self,video=False, gae=False, bdf=False,time=False,video_folders=["Sessions"]):
        if(video):
            if(not self.video_path):
                return False
            for folder in video_folders:
                if(not path.exists(self.video_path.replace("Sessions",folder))):
                    return False
        if(gae and type(self.gae_data)==type(None)):
            return False
        if(bdf and not (self.bdf_path and path.exists(self.bdf_path))):
            return False
        if(time and not self.duration):
            return False

        return True

class FFPP(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.category = 'train'
        C.root_dir = './datasets/ffpp/'
        C.detection_level = 'video'
        C.types = ['REAL', 'DF','F2F','FS','NT']
        C.compressions = ['raw']
        C.name = "FFPP"
        C.scale = 1.0
        C.pack = 0
        C.pair = 0
        C.contrast = 0
        return C

    def __init__(self, config,num_frames,clip_duration, transform=None, accelerator=None, split='train',index=0):
        assert 0 <= config.scale <= 1
        self.TYPE_DIRS = {
            'REAL': 'real/',
            # 'DFD' : 'data/original_sequences/actors/',
            'DF'  : 'DF/',
            'FS'  : 'FS/',
            'F2F' : 'F2F/',
            'NT'  : 'NT/',
            # 'FSH' : 'data/manipulated_sequences/FaceShifter/',
            # 'DFD-FAKE' : 'data/manipulated_sequences/DeepFakeDetection/',
        }
        self.category = config.category.lower()
        self.name = config.name.lower()
        self.root = path.expanduser(config.root_dir)
        self.detection_level = config.detection_level
        self.types = config.types
        self.compressions = config.compressions
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.split = split
        self.transform = transform
        self.index = index
        self.scale = config.scale
        self.pack = bool(config.pack)
        self.pair = bool(config.pair)
        self.contrast = bool(config.contrast)
        # available clips per data
        self.video_list = []

        # stacking data clips
        self.stack_video_clips = []

        self._build_video_table(accelerator)
        self._build_video_list(accelerator)
        
    def _build_video_table(self, accelerator):
        self.video_table = {}

        progress_bar = tqdm(self.types, disable=not accelerator.is_local_main_process)
        for df_type in progress_bar:
            self.video_table[df_type] = {}
            for comp in self.compressions:
                video_cache = path.expanduser(f'./.cache/dfd-clip/videos/{df_type}-{comp}.pkl')
                if path.isfile(video_cache):
                    with open(video_cache, 'rb') as f:
                        videos = pickle.load(f)
                    self.video_table[df_type][comp] = videos
                    continue

                # subdir
                subdir = path.join(self.root, self.TYPE_DIRS[df_type], f'{comp}/videos')

                video_metas =  {}

                # video table
                for f in  scandir(subdir):
                    if '.avi' in f.name:
                        cap = cv2.VideoCapture(f.path)
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        video_metas[f.name[:-4]] ={
                            "fps" : fps,
                            "frames" : frames,
                            "duration": frames/fps,
                            "path": f.path
                        }
                        cap.release()
                
                # description
                progress_bar.set_description(f"{df_type}: {comp}/videos")

                # caching
                if accelerator.is_local_main_process:
                    makedirs(path.dirname(video_cache), exist_ok=True)
                    with open(video_cache, 'wb') as f:
                        pickle.dump(video_metas, f)

                self.video_table[df_type][comp] = video_metas
        
    def _build_video_list(self, accelerator):
        self.video_list = []
        
        with open(path.join(self.root, 'splits', f'{self.split}.json')) as f:
            idxs = json.load(f)
            
        for df_type in self.types:
            for comp in self.compressions:
                comp_videos = []
                adj_idxs = [i for inner in idxs for i in inner] if df_type == 'REAL' else ['_'.join(idx) for idx in idxs] + ['_'.join(reversed(idx)) for idx in idxs]

                for idx in adj_idxs:
                    if idx in self.video_table[df_type][comp]:
                        clips = int(self.video_table[df_type][comp][idx]["duration"]//self.clip_duration)
                        if(clips > 0):
                            comp_videos.append((df_type, comp, idx, clips))
                    else:
                        accelerator.print(f'Warning: video {path.join(self.root, self.TYPE_DIRS[df_type], comp, "videos", idx)} does not present in the processed dataset.')
                self.video_list += comp_videos[:int(self.scale * len(comp_videos))]

        # stacking up the amount of data clips for further usage
        self.stack_video_clips = [0]
        for _,_,_,i in self.video_list:
            self.stack_video_clips.append(self.stack_video_clips[-1] + i)
        self.stack_video_clips.pop(0)

    def __len__(self):
        if(self.pack):
            return len(self.video_list)
        else:
            return self.stack_video_clips[-1]
    
    def __getitem__(self,idx):
        if(self.pack):
            start = 0 if idx == 0 else self.stack_video_clips[idx-1]
            end = self.stack_video_clips[idx]
            frames = []
            label = []
            mask = []
            for i in range(start,end):
                try:
                    result = self.get_dict(idx)
                except:
                    continue
                else:
                    for comp in result["frames"].keys():
                        frames.append(result["frames"][comp])
                        label.append(result["label"])
                        mask.append(result["mask"])
            return frames,label,mask,self.index
        elif(self.contrast):
            _, df_type, _, _, _ = self.__video_info(idx)
            main_label = (not df_type == "REAL")
            main_idx = idx
            auxi_idx = random.randint(0,len(self))
            result = []
            result.append(self.get_dict(main_idx,target_label=main_label))
            result.append(self.get_dict(auxi_idx,target_label=(not main_label)))
            return *[[ _r[name] for _r in result] for name in ["frames","label","mask"] ] , [self.index]*2
        else:
            result = self.get_dict(idx)
            return result["frames"],result["label"],result["mask"],self.index

    def get_dict(self,idx,block=False,target_label=None):
        while(True):
            try:
                # video_idx =  next(i for i,x in enumerate(self.stack_video_clips) if  idx < x)
                # df_type, comp, video_name, clips = self.video_list[video_idx]
                video_idx, df_type, comp, video_name, clips = self.__video_info(idx)

                # while specified the target label, resample a video index to match.
                if(not target_label == None and not target_label and not df_type=="REAL"):
                    idx = random.randrange(0,len(self))
                    continue

                video_meta = self.video_table[df_type][comp][video_name]
                video_offset_duration =  (idx - (0 if video_idx == 0 else self.stack_video_clips[video_idx-1]))*self.clip_duration
                logging.debug(f"Item/Video Index:{idx}/{video_idx}")
                logging.debug(f"Item DF/COMP:{df_type}/{comp}")

                # video frame processing
                frames = {}
                for target_comp in ["raw","c23"]:
                    _frames = []

                    vid_path = video_meta["path"]

                    if not target_comp in vid_path:
                        if not self.pair:
                            continue
                        else:
                            vid_path = vid_path.replace(comp,target_comp)
                    
                    
                    vid_reader = torchvision.io.VideoReader(
                        vid_path, 
                        "video"
                    )
                    # - frames per second
                    video_sample_freq = vid_reader.get_metadata()["video"]["fps"][0]
                    # - the amount of frames to skip
                    video_sample_offset = int(video_offset_duration)
                    # - the amount of frames for the duration of a clip
                    video_clip_samples = int(video_sample_freq * self.clip_duration)
                    # - the amount of frames to skip in order to meet the num_frames per clip.(excluding the head & tail frames )
                    video_sample_stride = ((video_clip_samples-1) / (self.num_frames - 1))/video_sample_freq
                    logging.debug(f"Loading Video: {video_meta['path']}")
                    logging.debug(f"Sample Offset: {video_sample_offset}")
                    logging.debug(f"Sample Stride: {video_sample_stride}")
                    # - fetch frames of clip duration
                    for sample_idx in range(self.num_frames):
                        try:
                            vid_reader.seek(video_sample_offset + sample_idx * video_sample_stride)
                            frame = next(vid_reader)
                            _frames.append(frame["data"])
                        except Exception as e:
                            raise Exception(f"unable to read video frame of sample index:{sample_idx}")
                    _frames = torch.stack(_frames)
                    del vid_reader

                    # transformation
                    if (self.transform):
                        _frames = self.transform(_frames)
                    
                    frames[target_comp] = _frames

                # padding and masking missing frames.
                mask = torch.tensor([1.] * len(frames[comp]) +
                                    [0.] * (self.num_frames - len(frames[comp])), dtype=torch.bool)
                
                for target_comp in ["raw","c23"]:
                    if target_comp in frames and len(frames[target_comp]) < self.num_frames:
                        diff = self.num_frames - len(frames[target_comp])
                        padding = torch.zeros((diff, *frames[target_comp].shape[1:]),dtype=torch.uint8)
                        frames[target_comp] = torch.concatenate((frames[target_comp], padding))
                
                return {
                    "frames":frames,
                    "label": 0 if df_type == "REAL" else 1,
                    "mask":mask,
                }
            except Exception as e:
                logging.error(f"Error occur: {e}")
                if block:
                    raise e
                else:
                    idx = random.randrange(0,len(self))

    def __video_info(self,idx):
        video_idx =  next(i for i,x in enumerate(self.stack_video_clips) if  idx < x)
        return video_idx, *self.video_list[video_idx]

    def collate_fn(self,batch):
        _frames,_label,_mask,_index = list(zip(*batch))
        
        if(self.contrast):
            _frames = [i for l in _frames for i in l]
            _label = [i for l in _label for i in l]
            _mask = [i for l in _mask for i in l]
            _index = [i for l in _index for i in l]
        
        _comps = list(_frames[0].keys())
        num_vids = len(_frames)
        num_comps = len(_comps)
        frames = []
        
        for _frame in _frames:
            for comp in _comps:
                frames.append(_frame[comp])
        
        frames = torch.stack(frames)
        label = torch.tensor(_label).repeat_interleave(num_comps,dim=0)
        mask = torch.stack(_mask).repeat_interleave(num_comps,dim=0)
        index = torch.tensor(_index).repeat_interleave(num_comps,dim=0)
        comps = _comps*num_vids
        
        return [frames,label,mask,comps,index]

class RPPG(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.category = 'train'
        C.root_dir = './datasets/hci/'
        C.detection_level = 'video'
        C.train_ratio = 0.95
        C.scale = 1.0
        C.cropped_folder="cropped_faces"
        C.meta_folder="Metas"
        C.measure_folder = "Measures"
        C.name="RPPG"
        C.compressions = ["raw"]
        C.runtime = True
        C.label_type = "dist"
        C.label_dim = 140
        return C

    def __init__(self, config,num_frames,clip_duration, transform=None, accelerator=None, split='train', index=0, save_meta=False):
        assert 0 <= config.scale <= 1, "config.scale out of range"
        assert 0 <= config.train_ratio <= 1, "config.train_ratio out of range"
        assert 140 <= config.label_dim , "config.label_dim should be atleast 140."
        assert split in ["train","val"], "split value not acceptable"
        assert config.label_type in ["num","dist"]
        self.category = config.category.lower()
        self.name = config.name.lower()
        # HCI datasets recorded videos with 61 fps
        self.transform = transform
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.index = index
        self.scale = config.scale
        self.compressions = config.compressions
        self.cropped_folder =  config.cropped_folder
        self.runtime = config.runtime
        self.label_type = config.label_type
        self.label_dim = config.label_dim
        # dataset consistency
        rng = random.Random()
        rng.seed(777)
        session_dirs = sorted(glob(path.join(config.root_dir,"Sessions","*")))
        rng.shuffle(session_dirs)

        # dataset splitting
        if split == "train":
            target_sessions = session_dirs[:int(len(session_dirs)*config.train_ratio*self.scale)]
        elif split == "val":
            target_sessions = session_dirs[int(len(session_dirs)*((1-config.train_ratio)*(1-self.scale) + config.train_ratio)):]
        
        # speed up dataset initialization
        if (not config.meta_folder):
            logging.info("Meta folder unspecified, building session meta infos....")
            self.session_metas = [
                SessionMeta(
                    session_dir
                ) 
                for session_dir in target_sessions
            ]
            if(save_meta):
                for meta in self.session_metas:
                    meta_dir = meta.session_dir.replace("Sessions","Metas")
                    makedirs(meta_dir,exist_ok=True)
                    with open(path.join(meta_dir,"meta.pickle"),"wb") as file:
                        pickle.dump(obj=meta,file=file)
        else:
            logging.info("Meta folder specified, loading meta infos....")
            self.session_metas = [None for _ in range(len(target_sessions))]
            for i,session_dir in tqdm(enumerate(target_sessions)):
                try:
                    with open(path.join(session_dir.replace("Sessions",config.meta_folder),"meta.pickle"),"rb") as file:
                        self.session_metas[i] = pickle.load(file)
                except Exception as e:
                    logging.debug(f"Error while loading meta pickle: '{e}'")

        logging.info("Session meta created.")
        logging.debug(f"Number of session metas before checking: {len(self.session_metas)}")
        # remove erroneous/missing datas.
        for comp in self.compressions:
            self.session_metas = [
                meta for meta in self.session_metas 
                if meta and meta.require_check(
                    video=True,
                    bdf=True,
                    time=True,
                    video_folders=[path.join(config.cropped_folder,comp)]
                )
            ]
        logging.info("Session meta filtered.")
        logging.debug(f"Current number of sessions: {len(self.session_metas)}")

        # load rppg heartrate measures.
        if(not self.runtime):
            _session_measures = []
            _session_metas = []

            for meta in self.session_metas:
                try:
                    with open(path.join(meta.session_dir.replace("Sessions","Measures"),"data.pickle"),"rb") as f:
                        _session_measures.append(
                            pickle.load(f)
                        )
                        _session_metas.append(meta)
                except Exception as e:
                    continue

            self.session_metas = _session_metas
            self.session_measures = _session_measures
            logging.info("Session measures loaded.")
            logging.debug(f"Current number of sessions: {len(self.session_metas)}")

        # total number of ready-to-use session metadata.
        logging.debug(f"Number of session metas ready for training: {len(self.session_metas)}")
        
        # calculate available clips per session
        self.session_clips = [int(meta.duration // self.clip_duration) for meta in self.session_metas]

        # stacking up the amount of session clips for further usage
        self.stack_session_clips = [0]
        for i in self.session_clips:
            self.stack_session_clips.append(self.stack_session_clips[-1] + i)
        self.stack_session_clips.pop(0)

    def __len__(self):
        return self.stack_session_clips[-1]*len(self.compressions)
    
    def __getitem__(self,idx):
        result = self.get_dict(idx,self.runtime)
        return result["frames"],result["label"],result["mask"],self.index
        
    def get_dict(self,idx,runtime=False):
        item_load_begin = time()
        while(True):
            try:
                comp = self.compressions[int(idx //self.stack_session_clips[-1])]
                idx = idx % self.stack_session_clips[-1]
                session_idx =  next(i for i,x in enumerate(self.stack_session_clips) if  idx < x)
                session_meta = self.session_metas[session_idx]
                session_offset_duration =  (idx - (0 if session_idx == 0 else self.stack_session_clips[session_idx-1]))*self.clip_duration
                hr_data = None
                measures = None
                wd = None
                logging.debug(f"Item/Session Index:{idx}/{session_idx}")

                # heart rate data processing
                rppg_load_begin = time()
                # - the ERG sample frequency
                hr_sample_freq =  session_meta.session_hr_sample_freq
                # - the amount of samples to skip, including the 30s stimulation offset and session clip offset.
                hr_sample_offset =  session_meta.flag_hr_beg_sample + int(session_offset_duration*hr_sample_freq)
                # - the amount of samples for the duration of a clip
                hr_clip_samples = int(hr_sample_freq*self.clip_duration)
                # - the end sample of the session clip
                hr_sample_end = hr_sample_offset + hr_clip_samples
                if(not runtime):
                    # - interpolate the hr sample
                    session_measure= self.session_measures[session_idx]
                    measure_idx =  next(i for i,x in enumerate(session_measure["idx"]) if  hr_sample_end <= x)
                    assert 0 < measure_idx <= len(session_measure["idx"]) , f"erroneous measure index {measure_idx} for end sample {hr_sample_end}"
                    # - calculate the distance ratio of the session clip to the two nearest preprocessed measure locations.
                    measure_ratio = (session_measure["idx"][measure_idx] - hr_sample_end) / (session_measure["idx"][measure_idx] - session_measure["idx"][measure_idx-1])
                    # - perform interpolation
                    bpm = (
                        measure_ratio  * session_measure["data"][measure_idx-1]["bpm"] +
                        (1 - measure_ratio ) * session_measure["data"][measure_idx]["bpm"] 
                    )
                else:
                    signals, signal_headers, _ = BDFReader.read_edf(session_meta.bdf_path,ch_names=["EXG1","EXG2","EXG3","Status"])
                    _hr_datas = []
                    for hr_channel_idx in range(3):
                        try:
                            assert  int(session_meta.session_hr_sample_freq) == int(signal_headers[hr_channel_idx]["sample_frequency"]), "heart rate frequency mismatch between metadata and the bdf file."
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

                # - heart rate validation
                assert 41 <= bpm <= 180, f"bpm located out of the defined range: {bpm}"
                # - create label
                if(self.label_type == "dist"):
                    label = torch.tensor([1/(pow(2*math.pi,0.5))*pow(math.e,(-pow((k-(bpm-41)),2)/2)) for k in range(self.label_dim)])
                elif(self.label_type == "num"):
                    label =  bpm - 41
                logging.debug(f"rPPG Load Duration:{time() - rppg_load_begin}")


                # video frame processing
                video_load_begin= time()
                frames = []
                comp_video_path = session_meta.video_path.replace(
                    "Sessions",
                    path.join("Sessions" if not self.cropped_folder else self.cropped_folder,comp)
                )
                vid_reader = torchvision.io.VideoReader(
                    comp_video_path, 
                    "video"
                )
                assert int(session_meta.session_video_sample_freq) == int(vid_reader.get_metadata()["video"]["fps"][0]), f"video sample frequency mismatch: {int(session_meta.session_video_sample_freq)},{int(vid_reader.get_metadata()['video']['fps'][0])}"
                video_sample_freq = session_meta.session_video_sample_freq
                # - the amount of frames to skip
                video_sample_offset = int(session_meta.flag_video_beg_sample - session_meta.session_video_beg_sample)/video_sample_freq + int(session_offset_duration)
                # - the amount of frames for the duration of a clip
                video_clip_samples = int(session_meta.session_video_sample_freq * self.clip_duration)
                # - the amount of frames to skip in order to meet the num_frames per clip.(excluding the head & tail frames )
                video_sample_stride = (video_clip_samples-1) / (self.num_frames - 1) / video_sample_freq
                # - fetch frames of clip duration
                logging.debug(f"Loading video: {comp_video_path}")
                logging.debug(f"Sample Offset: {video_sample_offset}")
                logging.debug(f"Sample Stride: {video_sample_stride}")
                # - fast forward to the the sampling start.
                for sample_idx in range(self.num_frames):
                    try:
                        vid_reader.seek(video_sample_offset +video_sample_stride*sample_idx)
                        frame = next(vid_reader)
                        frames.append(frame["data"])
                    except Exception as e:
                        raise Exception(f"unable to read video frame of sample index:{sample_idx}")
                frames = torch.stack(frames)
                del vid_reader
                logging.debug(f"Video Load Duration:{time() - video_load_begin}")

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
                
                logging.debug(f"Item Load Duration:{time() - item_load_begin}")
                return {
                    "frames":frames,
                    "label":label,
                    "mask":mask,
                    "hr_data": hr_data,
                    "wd":wd,
                    "measure":measures,
                }
            
            except Exception as e:
                logging.error(f"Error occur: {e}")
                logging.error(traceback.format_exc())
                idx = random.randrange(0,len(self))

class CDF(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.category = 'CDF'
        C.root_dir = './datasets/cdf/'
        C.name = "CDF"
        C.scale = 1.0
        C.pack = 0
        return C

    def __init__(self, config,num_frames,clip_duration, transform=None, accelerator=None, split="test", index=0,*args,**kwargs):
        if(not split == "test"):
            logging.warn(f"Dataset {self.__class__.__name__.upper()} currently support only the test split.")
            split = "test"
        assert 0 <= config.scale <= 1
        assert split == "test", f"Split '{split.upper()}' Not Implemented."

        self.category = config.category.lower()
        self.name = config.name.lower()
        self.root = path.expanduser(config.root_dir)
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.transform = transform
        self.index = index
        self.scale = config.scale
        self.pack = bool(config.pack)
        self.split = split
        # available clips per data
        self.video_list = []

        # stacking data clips
        self.stack_video_clips = []

        self._build_video_table(accelerator)
        self._build_video_list(accelerator)
        
    def _build_video_table(self, accelerator):
        self.video_table = {}

        progress_bar = tqdm(["REAL","FAKE"], disable=not accelerator.is_local_main_process)
        for label in progress_bar:
            self.video_table[label] = {}
            video_cache = path.expanduser(f'./.cache/dfd-clip/videos/{self.__class__.__name__}-{label}.pkl')
            if path.isfile(video_cache):
                with open(video_cache, 'rb') as f:
                    videos = pickle.load(f)
                self.video_table[label] = videos
                continue

            # subdir
            subdir = path.join(self.root, label, 'videos')

            video_metas =  {}

            # video table
            for f in  scandir(subdir):
                if '.avi' in f.name:
                    cap = cv2.VideoCapture(f.path)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_metas[f.name[:-4]] ={
                        "fps" : fps,
                        "frames" : frames,
                        "duration": frames/fps,
                        "path": f.path
                    }
                    cap.release()
            
            # description
            progress_bar.set_description(f"{label}/videos")

            # caching
            if accelerator.is_local_main_process:
                makedirs(path.dirname(video_cache), exist_ok=True)
                with open(video_cache, 'wb') as f:
                    pickle.dump(video_metas, f)

            self.video_table[label] = video_metas
        
    def _build_video_list(self, accelerator):
        self.video_list = []
        for label in ["REAL","FAKE"]:
            video_list = pd.read_csv(
                path.join(self.root, 'csv_files', f'{self.split}_{label.lower()}.csv'),
                sep=' ',
                header=None,
                names=["name","label"]
            )
            
            _videos = []

            for filename in video_list["name"]:
                name,ext =  os.path.splitext(filename)
                if name in self.video_table[label]:
                    clips = int(self.video_table[label][name]["duration"]//self.clip_duration)
                    if(clips > 0):
                        _videos.append((label.upper(), name, clips))
                else:
                    accelerator.print(f'Warning: video {path.join(self.root, label, "videos", name)} does not present in the processed dataset.')
            self.video_list += _videos[:int(self.scale * len(_videos))]

        # stacking up the amount of data clips for further usage
        self.stack_video_clips = [0]
        for _,_,i in self.video_list:
            self.stack_video_clips.append(self.stack_video_clips[-1] + i)
        self.stack_video_clips.pop(0)

    def __len__(self):
        if(self.pack):
            return len(self.video_list)
        else:
            return self.stack_video_clips[-1]

    def __getitem__(self,idx):
        if(self.pack):
            start = 0 if idx == 0 else self.stack_video_clips[idx-1]
            end = self.stack_video_clips[idx]
            frames = []
            label = []
            mask = []
            for i in range(start,end):
                try:
                    result = self.get_dict(idx)
                except:
                    continue
                else:
                    frames.append(result["frames"])
                    label.append(result["label"])
                    mask.append(result["mask"])
            return frames,label,mask,self.index
        else:
            result = self.get_dict(idx)
            return result["frames"],result["label"],result["mask"],self.index

    def get_dict(self,idx,block=False):
        while(True):
            try:
                video_idx =  next(i for i,x in enumerate(self.stack_video_clips) if  idx < x)
                label, video_name, clips = self.video_list[video_idx]
                video_meta = self.video_table[label][video_name]
                video_offset_duration =  (idx - (0 if video_idx == 0 else self.stack_video_clips[video_idx-1]))*self.clip_duration
                logging.debug(f"Item/Video Index:{idx}/{video_idx}")
                logging.debug(f"Item Label:{label}")

                # video frame processing
                frames = []
                vid_reader = torchvision.io.VideoReader(
                    video_meta["path"], 
                    "video"
                )
                # - frames per second
                video_sample_freq = vid_reader.get_metadata()["video"]["fps"][0]
                # - the amount of frames to skip
                video_sample_offset = int(video_offset_duration)
                # - the amount of frames for the duration of a clip
                video_clip_samples = int(video_sample_freq * self.clip_duration)
                # - the amount of frames to skip in order to meet the num_frames per clip.(excluding the head & tail frames )
                video_sample_stride = ((video_clip_samples-1) / (self.num_frames - 1))/video_sample_freq
                logging.debug(f"Loading Video: {video_meta['path']}")
                logging.debug(f"Sample Offset: {video_sample_offset}")
                logging.debug(f"Sample Stride: {video_sample_stride}")
                # - fetch frames of clip duration
                for sample_idx in range(self.num_frames):
                    try:
                        vid_reader.seek(video_sample_offset + sample_idx * video_sample_stride)
                        frame = next(vid_reader)
                        frames.append(frame["data"])
                    except Exception as e:
                        raise Exception(f"unable to read video frame of sample index:{sample_idx}")
                frames = torch.stack(frames)
                del vid_reader

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
                    "label": 0 if label == "REAL" else 1,
                    "mask":mask,
                }
            except Exception as e:
                logging.error(f"Error occur: {e}")
                if(block):
                    raise e
                else:
                    idx = random.randrange(0,len(self))

    def collate_fn(self,batch):
        return default_collate(batch)
    
class DFDC(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.category = 'DFDC'
        C.root_dir = './datasets/dfdc/'
        C.name = "DFDC"
        C.scale = 1.0
        C.pack = 0
        return C

    def __init__(self, config,num_frames,clip_duration, transform=None, accelerator=None, split="test",index=0,*args,**kwargs):
        if(not split == "test"):
            logging.warn(f"Dataset {self.__class__.__name__.upper()} currently support only the test split.")
            split = "test"
        assert 0 <= config.scale <= 1
        assert split == "test", f"Split '{split.upper()}' Not Implemented."

        self.category = config.category.lower()
        self.name = config.name.lower()
        self.root = path.expanduser(config.root_dir)
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.transform = transform
        self.index = index
        self.scale = config.scale
        self.pack = bool(config.pack)
        self.split = split
        # available clips per data
        self.video_list = []

        # stacking data clips
        self.stack_video_clips = []

        self._build_video_table(accelerator)
        self._build_video_list(accelerator)
        
    def _build_video_table(self, accelerator):
        self.video_table = {}

        video_cache = path.expanduser(f'./.cache/dfd-clip/videos/{self.__class__.__name__}.pkl')
        if path.isfile(video_cache):
            with open(video_cache, 'rb') as f:
                videos = pickle.load(f)
            self.video_table = videos
        else:
            # subdir
            subdir = path.join(self.root, 'videos')

            video_metas =  {}

            # video table
            for f in  scandir(subdir):
                if '.avi' in f.name:
                    cap = cv2.VideoCapture(f.path)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_metas[f.name[:-4]] ={
                        "fps" : fps,
                        "frames" : frames,
                        "duration": frames/fps,
                        "path": f.path
                    }
                    cap.release()

            # caching
            if accelerator.is_local_main_process:
                makedirs(path.dirname(video_cache), exist_ok=True)
                with open(video_cache, 'wb') as f:
                    pickle.dump(video_metas, f)

            self.video_table = video_metas
        
    def _build_video_list(self, accelerator):
        self.video_list = []

        video_list = pd.read_csv(
            path.join(self.root, 'csv_files', f'{self.split}.csv'),
            sep=' ',
            header=None,
            names=["name","label"]
        )
        
        _videos = []

        for index, row in video_list.iterrows():
            filename = row["name"]
            name,ext =  os.path.splitext(filename)
            label = "REAL" if row["label"] == 0 else "FAKE"
            if name in self.video_table:
                clips = int(self.video_table[name]["duration"]//self.clip_duration)
                if(clips > 0):
                    _videos.append((label, name, clips))
            else:
                accelerator.print(f'Warning: video {path.join(self.root, label, "videos", name)} does not present in the processed dataset.')
        self.video_list += _videos[:int(self.scale * len(_videos))]

        # stacking up the amount of data clips for further usage
        self.stack_video_clips = [0]
        for _,_,i in self.video_list:
            self.stack_video_clips.append(self.stack_video_clips[-1] + i)
        self.stack_video_clips.pop(0)

    def __len__(self):
        if(self.pack):
            return len(self.video_list)
        else:
            return self.stack_video_clips[-1]
    
    def __getitem__(self,idx):
        if(self.pack):
            start = 0 if idx == 0 else self.stack_video_clips[idx-1]
            end = self.stack_video_clips[idx]
            frames = []
            label = []
            mask = []
            for i in range(start,end):
                try:
                    result = self.get_dict(idx)
                except:
                    continue
                else:
                    frames.append(result["frames"])
                    label.append(result["label"])
                    mask.append(result["mask"])
            return frames,label,mask,self.index
        else:
            result = self.get_dict(idx)
            return result["frames"],result["label"],result["mask"],self.index

    def get_dict(self,idx,block=False):
        while(True):
            try:
                video_idx =  next(i for i,x in enumerate(self.stack_video_clips) if  idx < x)
                label, video_name, clips = self.video_list[video_idx]
                video_meta = self.video_table[video_name]
                video_offset_duration =  (idx - (0 if video_idx == 0 else self.stack_video_clips[video_idx-1]))*self.clip_duration
                logging.debug(f"Item/Video Index:{idx}/{video_idx}")
                logging.debug(f"Item Label:{label}")

                # video frame processing
                frames = []
                vid_reader = torchvision.io.VideoReader(
                    video_meta["path"], 
                    "video"
                )
                # - frames per second
                video_sample_freq = vid_reader.get_metadata()["video"]["fps"][0]
                # - the amount of frames to skip
                video_sample_offset = int(video_offset_duration)
                # - the amount of frames for the duration of a clip
                video_clip_samples = int(video_sample_freq * self.clip_duration)
                # - the amount of frames to skip in order to meet the num_frames per clip.(excluding the head & tail frames )
                video_sample_stride = ((video_clip_samples-1) / (self.num_frames - 1))/video_sample_freq
                logging.debug(f"Loading Video: {video_meta['path']}")
                logging.debug(f"Sample Offset: {video_sample_offset}")
                logging.debug(f"Sample Stride: {video_sample_stride}")
                # - fetch frames of clip duration
                for sample_idx in range(self.num_frames):
                    try:
                        vid_reader.seek(video_sample_offset + sample_idx * video_sample_stride)
                        frame = next(vid_reader)
                        frames.append(frame["data"])
                    except Exception as e:
                        raise Exception(f"unable to read video frame of sample index:{sample_idx}")
                frames = torch.stack(frames)
                del vid_reader

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
                    "label": 0 if label == "REAL" else 1,
                    "mask":mask,
                }
            except Exception as e:
                logging.error(f"Error occur: {e}")
                if(block):
                    raise e
                else:
                    idx = random.randrange(0,len(self))

    def collate_fn(self,batch):
        return default_collate(batch)

