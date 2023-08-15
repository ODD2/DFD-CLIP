from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob
import os
import pandas as pd


def init_ff(phase, raw=False, level='frame', n_frames=8):
    dataset_path = ""
    if raw:
        dataset_path = '/app/data/FaceForensicsPPRaw/'
    else:
        dataset_path = '/app/data/FaceForensicsPP/'

    image_list = []
    label_list = []

    folder_list = sorted(glob(os.path.join(dataset_path, 'frames/*')))
    filelist = []
    list_dict = json.load(open(os.path.join(dataset_path, f'{phase}.json'), 'r'))
    for i in list_dict:
        filelist += i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
    if level == 'video':
        label_list = [0] * len(folder_list)
        return folder_list, label_list
    else:
        for i in range(len(folder_list)):
            images_temp = sorted(glob(folder_list[i] + '/*.png'))
            if n_frames < len(images_temp):
                images_temp = [images_temp[round(i)] for i in np.linspace(0, len(images_temp) - 1, n_frames)]
            image_list += images_temp
            label_list += [0] * len(images_temp)
    print(dataset_path, len(image_list))
    return image_list, label_list


# def init_ff(phase,level='frame',n_frames=8):
#     dataset_path='/app/data/FaceForensics++/original_sequences/youtube/raw/frames/'


#     image_list=[]
#     label_list=[]


#     folder_list = sorted(glob(dataset_path+'*'))
#     filelist = []
#     list_dict = json.load(open(f'/app/data/FaceForensics++/{phase}.json','r'))
#     for i in list_dict:
#         filelist+=i
#     folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

#     if level =='video':
#         label_list=[0]*len(folder_list)
#         return folder_list,label_list
#     for i in range(len(folder_list)):
#         # images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
#         images_temp=sorted(glob(folder_list[i]+'/*.png'))
#         if n_frames<len(images_temp):
#             images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
#         image_list+=images_temp
#         label_list+=[0]*len(images_temp)

#     return image_list,label_list
