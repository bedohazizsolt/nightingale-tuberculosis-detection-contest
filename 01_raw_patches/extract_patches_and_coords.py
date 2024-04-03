"""
Copyright 2024 Zsolt Bedőházi, András M. Biricz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import sys
import os
import random
import glob
import h5py
import cv2
import yaml
import torch
import matplotlib
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import DictConfig

        
        
def extract_patches_and_coords(image, patch_size, padding_value, tb_positive, save_name):
    
    height, width, channels = image.shape

    # Calculate the number of patches in each dimension
    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))
    
    padded_image = torch.zeros((rows*patch_size, cols*patch_size, channels), dtype=torch.uint8)
    padded_image[:image.shape[0], :image.shape[1], :] = torch.from_numpy(image)
    patches = padded_image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.reshape(-1, 3, patch_size, patch_size)
    patches = np.moveaxis(np.array(patches),1,3) # (B,224,224,3)
    
    coords = np.array([(x * patch_size, y * patch_size) for y in range(rows) for x in range(cols)])
    
    # save to an H5 file
    with h5py.File(save_name, 'w') as h5f:
        h5f.create_dataset("imgs", data=patches)
        h5f.create_dataset("coords", data=coords)
        h5f.create_dataset("tb_positive", data=tb_positive)
        
        
        
if __name__ == "__main__":
    
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxnum_threads", type=int, required=False, default=0)
    parser.add_argument("--num", type=int, required=False, default=1)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config_level_0", type=str, required=True)
    parser.add_argument("--config_level_1", type=str, required=True)
    
    # get the arguments parsed
    args = parser.parse_args()
    maxnum_threads = args.maxnum_threads
    num = args.num
    dataset = args.dataset
    config_level_0 = args.config_level_0
    config_level_1 = args.config_level_1
    
    # Load config
    preproc_conf = OmegaConf.load("/home/ngsci/project/tuberculosis_detection/conf/preproc.yaml")
    preproc_conf = preproc_conf[config_level_0][config_level_1]
    print(dir(preproc_conf))
    data_root_dir = preproc_conf["data_root_dir"]
    if dataset == "contest":
        tb_labels = pd.read_csv(preproc_conf["tb_labels_csv"])
        dest_dir = preproc_conf["patch_dir"]
        print(dest_dir)
    elif dataset == "holdout":
        tb_labels = pd.read_csv(preproc_conf["tb_labels_csv_holdout"])
        tb_labels["tb_positive"] = -1*np.ones(tb_labels.shape[0])
        dest_dir = preproc_conf["patch_dir_holdout"]
    else:
        raise Exception("Invalid dataset! Use contest or holdout!")
    os.makedirs(dest_dir, exist_ok=True)
        
    print("Processing dataset: ", dataset)        
    
    padding_value = preproc_conf["padding_value"]
    patch_size = preproc_conf["patch_size"]
    
    
    # split to threads
    split_intervals = np.append(np.arange(0, tb_labels.shape[0], tb_labels.shape[0]/maxnum_threads).astype(int), tb_labels.shape[0])
    idxs = np.vstack( (split_intervals[:-1], split_intervals[1:]) ).T
    
    idx_to_run_now = idxs[num]
    print("Processing: ", idx_to_run_now)
    
        
    for idx in tqdm(range(idx_to_run_now[0], idx_to_run_now[1])):
        #load image
        image = np.array( Image.open(tb_labels.iloc[idx].file_path).convert('RGB'))
        save_name = dest_dir + os.path.basename(tb_labels.iloc[idx].file_path).replace(".jpg", ".h5")
      
        extract_patches_and_coords(image, patch_size, padding_value, tb_labels.iloc[idx].tb_positive, save_name)
    
