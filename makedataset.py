# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:00:46 2020

@author: Administrator
"""

import os
import os.path
import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
import argparse
from PIL import Image
class Dataset(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, file, trainrgb=True,trainsyn = True, shuffle=False):
		super(Dataset, self).__init__()
		self.trainrgb = trainrgb
		self.trainsyn = trainsyn
		self.train_haze	 = file
		
		h5f = h5py.File(self.train_haze, 'r')
		  
		self.keys = list(h5f.keys())
		if shuffle:
			random.shuffle(self.keys)
		h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):

		h5f = h5py.File(self.train_haze, 'r')
		  
		key = self.keys[index]
		data = np.array(h5f[key])
		h5f.close()
		return torch.Tensor(data)

def data_augmentation(clear, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    clear = np.transpose(clear, (2, 3, 0, 1))
    if mode == 0:
        # original
        clear = clear
    elif mode == 1:
        # flip up and down
        clear = np.flipud(clear)
    elif mode == 2:
        # rotate counterwise 90 degree
        clear = np.rot90(clear)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        clear = np.rot90(clear)
        clear = np.flipud(clear)
    elif mode == 4:
        # rotate 180 degree
        clear = np.rot90(clear, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        clear = np.rot90(clear, k=2)
        clear = np.flipud(clear)
    elif mode == 6:
        # rotate 270 degree
        clear = np.rot90(clear, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        clear = np.rot90(clear, k=3)
        clear = np.flipud(clear)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(clear, (2, 3, 0, 1))

def img_to_patches(img,win,stride,Syn=True):
    typ, chl, raw, col = img.shape
    chl = int(chl)
    num_raw = np.ceil((raw-win)/stride+1).astype(np.uint8)
    num_col = np.ceil((col-win)/stride+1).astype(np.uint8)
    count = 0
    total_process = int(num_col)*int(num_raw)
    img_patches = np.zeros([typ, chl, win, win, total_process])
    if Syn:
        for i in range(num_raw):
            for j in range(num_col):
                if stride * i + win <= raw and stride * j + win <=col:
                    img_patches[:,:,:,:,count] = img[:, :, stride*i : stride*i + win, stride*j : stride*j + win]
                elif stride * i + win > raw and stride * j + win<=col:
                    img_patches[:,:,:,:,count] = img[:, :,raw-win : raw,stride * j : stride * j + win]
                elif stride * i + win <= raw and stride*j + win>col:
                    img_patches[:,:,:,:,count] = img[:, :,stride*i : stride*i + win, col-win : col]
                else:
                    img_patches[:,:,:,:,count] = img[:, :,raw-win : raw,col-win : col]
                img_patches[:,:,:,:,count] = data_augmentation(img_patches[:, :, :, :, count], np.random.randint(0, 7))
                count +=1
    return img_patches

def read_img(img):
    return np.array(Image.open(img))/255.

# def Train_data(args):
#     file_list = os.listdir(f'{args.train_path}/{args.gt_name}')

#     with h5py.File(args.data_name, 'w') as h5f:
#         count = 0
#         for i in range(len(file_list)):
#             print(file_list[i])
#             img_list = []
            
#             img_list.append(read_img(f'{args.train_path}/{args.gt_name}/{file_list[i]}'))
#             for j in args.degradation_name:
#                 img_list.append(read_img(f'{args.train_path}/{j}/{file_list[i]}'))
            
#             img = np.stack(img_list,0)
#             img = img_to_patches(img.transpose(0, 3, 1, 2), args.patch_size, args.stride)
            
#             for nx in range(img.shape[4]):
#                 data = img[:,:,:,:,nx]
#                 print(count, data.shape)
#                 if args.compression:
#                     h5f.create_dataset(str(count), data=data, compression=args.compression, 
#                                      compression_opts=args.compression_opts)
#                 else:
#                     h5f.create_dataset(str(count), data=data)
#                 count += 1
#         h5f.close()

def Train_data(args):
    # Define exact ordering to match training code expectations
    ordered_types = [
        'blur',          # 0
        'blur_haze',     # 1
        'clear',         # 2  # Clear must be at index 2
        'haze',          # 3
        'haze_rain',     # 4
        'haze_snow',     # 5
        'low',           # 6
        'low_haze',      # 7
        'low_haze_rain', # 8
        'low_haze_snow', # 9
        'low_rain',      # 10
        'low_snow',      # 11
        'rain',          # 12
        'snow'           # 13
    ]
    
    
    file_list = os.listdir(f'{args.train_path}/clear')
    print(f"Total number of images: {len(file_list)}")

    with h5py.File(args.data_name, 'w') as h5f:
        count = 0
        for i in range(len(file_list)):
            print(f"Processing image {i+1}/{len(file_list)}: {file_list[i]}")
            img_list = []
            
            # Load images in exact order
            for type_name in ordered_types:
                if type_name == 'clear':
                    path = f'{args.train_path}/clear/{file_list[i]}'
                else:
                    path = f'{args.train_path}/{type_name}/{file_list[i]}'
                
                try:
                    img = read_img(path)
                    img_list.append(img)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    raise
            
            # Stack and process
            img = np.stack(img_list, 0)
            patches = img_to_patches(img.transpose(0, 3, 1, 2), 
                                   args.patch_size, 
                                   args.stride)
            
            # Save patches with metadata
            for nx in range(patches.shape[4]):
                data = patches[:,:,:,:,nx]
                try:
                    dset = h5f.create_dataset(
                        str(count), 
                        data=data,
                        compression='gzip',      # Use gzip compression
                        compression_opts=7,      # 7/9 compression level
                        chunks=True,            # Enable chunking for better compression
                        shuffle=True            # Enable shuffle filter for better compression
                    )
                    dset.attrs['image_source'] = file_list[i]
                    dset.attrs['degradation_order'] = ordered_types
                    count += 1
                except OSError as e:
                    print(f"Error saving patch: {e}")
                    raise
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} images, created {count} patches")
                h5f.flush()  # Force write to disk
   
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description = "Building the training patch database")
    parser.add_argument("--patch-size", type = int, default=256, help="Patch size")
    parser.add_argument("--stride", type = int, default=200, help="Size of stride")

    parser.add_argument("--train-path", type = str, default='./data/CDD-11_train', help="Train path")
    parser.add_argument("--data-name", type = str, default='dataset.h5', help="Data name")

    parser.add_argument("--gt-name", type = str, default='clear', help="HQ name")
    # parser.add_argument("--degradation-name", type = list, default=['low','haze','rain','snow',\
    # 'low_haze','low_rain','low_snow','haze_rain','haze_snow','low_haze_rain','low_haze_snow'], help="LQ name")
    parser.add_argument("--degradation-name", type=list, default=[
    'low', 'haze', 'rain', 'snow', 'blur',  # added 'blur'
    'low_haze', 'low_rain', 'low_snow',
    'haze_rain', 'haze_snow', 'blur_haze',  # added 'blur_haze'
    'low_haze_rain', 'low_haze_snow'
    ], help="LQ name")

    args = parser.parse_args()
    
    Train_data(args)