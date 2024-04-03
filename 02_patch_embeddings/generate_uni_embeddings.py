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


# Request the UNI model weights at: https://huggingface.co/MahmoodLab/UNI

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
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
import warnings
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download
import matplotlib.pyplot as plt
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


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
    
    
    # Load UNI model
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0
    )

    model.load_state_dict(torch.load("/home/ngsci/project/tuberculosis_detection/02_patch_embeddings/uni_model/pytorch_model.bin", map_location="cpu"), strict=True)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval();
    device = torch.device(f"cuda:{cuda_nr}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    
    print("Processing dataset: ", dataset)
    for idx in tqdm(range(idx_to_run_now[0], idx_to_run_now[1])):
        
        f = patch_dir + os.path.basename(tb_labels.iloc[idx].file_path).replace(".jpg", ".h5")
        
        file_name_w_ext = f.split('/')[-1]
        label = tb_labels.iloc[idx].tb_positive

        if not os.path.exists(dest_dir + file_name_w_ext):

            features_all = []
            coords_all = []

            # Open the HDF5 file
            #print(f)
            with h5py.File(f, 'r') as h5_file:

                # Get the total number of images in the dataset
                num_images = h5_file['imgs'].shape[0]
                #print('NUM IMGs:', num_images )

                # Set the number of parts to split the dataset into
                num_parts = 1 # 80

                # Set the size of each part
                part_size = num_images // num_parts

                # Process each part of the dataset
                for i in range(num_parts):

                    # Define the offset and count arguments for the current part
                    offset = i * part_size
                    count = part_size if i < num_parts-1 else num_images - offset

                    # Read the current part of the dataset into memory
                    imgs = h5_file['imgs'][offset:offset+count]
                    #print('PRINT:', i, imgs.shape, offset, offset+count)
                    
                     # Create a torch tensor for the images
                    imgs_tensor = torch.zeros((imgs.shape[0], imgs.shape[3], imgs.shape[2], imgs.shape[1]), dtype=torch.float32)
                    print(imgs_tensor.shape)

                    # Apply transforms to images
                    for i in range(imgs.shape[0]):
                        imgs_tensor[i] = transform(Image.fromarray(imgs[i].astype('uint8'), 'RGB')).unsqueeze(dim=0)

                    with torch.inference_mode():
                        inputs = imgs_tensor.to(device)
                        print("inputs shape: ", inputs.shape)
                    
                        outputs = model(inputs)
                        print("outputs shape: ", outputs.shape)

                        features_all.append(outputs.cpu().numpy())

                    # Get the coordinates for the current part of the dataset
                    coords_part = h5_file['coords'][offset:offset+count]
                    coords_all.append(coords_part)

                # Concatenate the embeddings and coordinates for all parts of the dataset
                features_all = np.concatenate(features_all)
                coords_all = np.concatenate(coords_all)
                
                print('Final shapes:', features_all.shape, coords_all.shape)

                # Save the embeddings and coordinates to a new HDF5 file
                filename = dest_dir + file_name_w_ext
                save_h5_file(filename, coords_all, features_all, label)
                
        

        else:
            #print(f)
            #print('Already done')
            pass
        
    