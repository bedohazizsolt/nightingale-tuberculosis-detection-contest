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


# Model weights available at: https://huggingface.co/facebook/dinov2-small

import sys
import os
import random
import glob
import h5py
import cv2
import yaml
import matplotlib
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
import warnings

from transformers import AutoImageProcessor, AutoModel


# Function to save data to HDF5
def save_h5_file(filename, coords, features, label):
    with h5py.File(filename, "w") as g:
        g.create_dataset("coords", data=coords)
        g.create_dataset("features", data=features)
        g.create_dataset("tb_positive", data=label)


if __name__ == "__main__":
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxnum_threads", type=int, required=False, default=1)
    parser.add_argument("--num", type=int, required=False, default=0)
    parser.add_argument("--cuda_nr", type=int, required=False, default=0)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config_level_0", type=str, required=True)
    parser.add_argument("--config_level_1", type=str, required=True)
    args = parser.parse_args()

    maxnum_threads = args.maxnum_threads
    num = args.num
    cuda_nr = args.cuda_nr
    dataset = args.dataset
    config_level_0 = args.config_level_0
    config_level_1 = args.config_level_1
    
    # Load config
    preproc_conf = OmegaConf.load("/home/ngsci/project/tuberculosis_detection/conf/preproc.yaml") 
    preproc_conf = preproc_conf[config_level_0][config_level_1]
    
    # Load dataset specific configurations
    if dataset == "contest":
        data_root_dir = preproc_conf["data_root_dir"]
        tb_labels = pd.read_csv(preproc_conf["tb_labels_csv"])
        patch_dir = preproc_conf["patch_dir"]
        dest_dir = preproc_conf["emb_dir"]
    elif dataset == "holdout":
        data_root_dir = preproc_conf["data_root_dir_holdout"]
        tb_labels = pd.read_csv(preproc_conf["tb_labels_csv_holdout"])
        tb_labels["tb_positive"] = -1 * np.ones(tb_labels.shape[0])
        patch_dir = preproc_conf["patch_dir_holdout"]
        dest_dir = preproc_conf["emb_dir_holdout"]
    else:
        raise Exception("Invalid dataset! Use contest or holdout!")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    patches_glob = os.path.join(patch_dir, f'*.h5')
    files = np.array( sorted( glob.glob(patches_glob) ) )
    print(f'Found {files.shape[0]} files')
    
    # split to threads
    split_intervals = np.append(np.arange(0, tb_labels.shape[0], tb_labels.shape[0]/maxnum_threads).astype(int), tb_labels.shape[0])
    idxs = np.vstack( (split_intervals[:-1], split_intervals[1:]) ).T
    
    idx_to_run_now = idxs[num]
    print("Processing: ", idx_to_run_now)
    
    # LOAD ViT-256 embedder
    device = torch.device(f"cuda:{cuda_nr}" if torch.cuda.is_available() else "cpu")
    
    # dinov2 small
    model_base_path = '/home/ngsci/project/tuberculosis_detection/02_patch_embeddings/vit_models/'
    model_load_path = model_base_path+"dinov2_small_model"
    processor_load_path = model_base_path+"dinov2_small_processor"

    # Load model and processor
    model = AutoModel.from_pretrained(model_load_path)
    processor = AutoImageProcessor.from_pretrained(processor_load_path)
    model.to(device)
    
    
    print("Processing dataset: ", dataset)
    for idx in tqdm(range(idx_to_run_now[0], idx_to_run_now[1])):
        
        f = patch_dir + os.path.basename(tb_labels.iloc[idx].file_path).replace(".jpg", ".h5")
        
        file_name_w_ext = f.split('/')[-1]
        label = tb_labels.iloc[idx].tb_positive

        if not os.path.exists(dest_dir + file_name_w_ext):

            features_all = []
            coords_all = []

            # Open the HDF5 file
            with h5py.File(f, 'r') as h5_file:

                num_images = h5_file['imgs'].shape[0]

                num_parts = 1 

                part_size = num_images // num_parts

                # Process each part of the dataset
                for i in range(num_parts):

                    offset = i * part_size
                    count = part_size if i < num_parts-1 else num_images - offset

                    imgs = h5_file['imgs'][offset:offset+count]

                    with torch.no_grad():
                        inputs = processor(images=imgs, return_tensors="pt")
                        inputs['pixel_values'] = inputs['pixel_values'].to(device)
                        
                        outputs = model(**inputs)
                        last_hidden_state_batch = outputs.last_hidden_state
                        cls_token_batch = last_hidden_state_batch[:,0]

                        features_all.append(cls_token_batch.cpu().numpy())

                    # Get the coordinates for the current part of the dataset
                    coords_part = h5_file['coords'][offset:offset+count]
                    coords_all.append(coords_part)

                # Concatenate the embeddings and coordinates for all parts of the dataset
                features_all = np.concatenate(features_all)
                coords_all = np.concatenate(coords_all)
                

                # Save the embeddings and coordinates to a new HDF5 file
                filename = dest_dir + file_name_w_ext
                save_h5_file(filename, coords_all, features_all, label)

        else:
            print(f)
            print('Already done')
            pass
    