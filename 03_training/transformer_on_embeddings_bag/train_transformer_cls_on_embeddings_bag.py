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

# import packages
import sys
import os
import random
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import auc as calc_auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from omegaconf import OmegaConf
from omegaconf import DictConfig
import argparse
from transformer_model_cls import ClsTokenTransformerClassifier


# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=False, default=0)
parser.add_argument("--config_level_0", type=str, required=True)
parser.add_argument("--preproc_config_level_1", type=str, required=True)
parser.add_argument("--train_config_level_1", type=str, required=True)
parser.add_argument("--splits", type=str, required=True)
parser.add_argument("--sampling", type=str, required=False, default='balanced')
parser.add_argument("--optimizer", type=str, required=False, default='AdamW')
parser.add_argument("--cuda_nr", type=int, required=False, default=0)



# get the arguments parsed
args = parser.parse_args()
FOLD = args.fold
config_level_0 = args.config_level_0
preproc_config_level_1 = args.preproc_config_level_1
train_config_level_1 = args.train_config_level_1
splits = args.splits
sampling = args.sampling
optimizer_name = args.optimizer
current_gpu = args.cuda_nr


# Load configs
preproc_conf = OmegaConf.load("../../conf/preproc.yaml")
train_conf = OmegaConf.load("../../conf/train.yaml")


# get current parts of configs
conf_preproc = preproc_conf[config_level_0][preproc_config_level_1]
conf_train = train_conf[config_level_0][train_config_level_1]

print(dir(preproc_conf))


print(conf_train["results_dir"])


## FUNCTIONS ##

def create_balanced_bag_subset(labels, minority_class_ratio=0.2, rnd_seed=38):
    # set random seed as given
    np.random.seed(rnd_seed)
    test_local_idx = []
    class_occurence = np.array(list(dict( Counter(labels) ).values()))[ np.argsort(list(dict( Counter(labels) ).keys()))]
    
    # calc class weights
    
    if sampling == 'sqrt': 
        sqrt_class_occurence = np.sqrt(class_occurence)
        class_weights = (sqrt_class_occurence / sqrt_class_occurence.sum()).astype(np.float32)
    
    elif sampling == 'balanced':
        class_weights = ( class_occurence / class_occurence.sum() ).astype(np.float32)
    
    else:
        print('Choose valid sampling method to deal with class imbalance!')
    
    class_weights_dict = dict( zip( np.arange(class_weights.shape[0]), class_weights ))
   
    nr_class_test = int(labels.shape[0]*np.min(class_weights)*minority_class_ratio)

    for s in np.unique(labels): #loop over labels
        s_idx = np.arange(labels.shape[0])[labels == s]
        rnd_idx = np.random.permutation(s_idx.shape[0])
        test_local_idx.append(s_idx[rnd_idx[:nr_class_test]])

    # aggregate all the balanced subset's indices
    test_idx = np.concatenate(test_local_idx)
    
    random.Random(23).shuffle(test_idx) # shuffle otherwise lables are ordered
    
    # other indices not in balanced set will be the rest
    train_idx = np.arange(labels.shape[0])[~np.in1d(np.arange(labels.shape[0]), test_idx)]
    
    return train_idx, test_idx
    

class h5_Dataset(Dataset):
    def __init__(self, emb_file, cv_df, transform=None):
        self.transform = transform
        self.file = emb_file
        self.cv_df = cv_df
        self.dataset_name = 'features'
        self.cv_samples_index = np.array([ int(os.path.basename(f).replace(".jpg","").replace("tb",""))-1 for f in self.cv_df["file_path"] ])
        self.cv_idx_to_all_idx = dict(zip(np.arange(self.cv_samples_index.shape[0]), self.cv_samples_index))
    
    def __len__(self):
        return len(self.cv_samples_index)

    def __getitem__(self, idx):

        all_idx = self.cv_idx_to_all_idx[idx]
        # Open the file and get the image data
        with h5py.File(self.file, 'r') as in_h5:

            # Get the image data from the file
            image_data = in_h5[self.dataset_name][all_idx]
            
            label = in_h5["tb_positive"][all_idx]
            
            if self.transform:
                image_data = self.transform(image_data)
        
        return image_data, label
        

def train_loop(cur, y_train_all, y_val_all, results_dir, 
               batch_size, num_epochs, model, n_classes, loss_fn=None, gc=32):  
            
    device=torch.device(f"cuda:{current_gpu}" if torch.cuda.is_available() else "cpu") 
    
    print('\nInit optimizer ...', end=' ')
    weight_decay = conf_train.weight_decay
    
    if conf_train["use_scheduler"]:
        print("\nUsing lr scheduler...")
        
        if optimizer_name == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=conf_train.initial_lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=conf_train.eta_min)
        
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf_train.initial_lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=conf_train.eta_min)
        
    else:
        
        if optimizer_name == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=conf_train.lr, weight_decay=weight_decay)
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf_train.lr, weight_decay=weight_decay)
        
        
    print('Done!')
    
    train_loss_all_epoch = []
    val_loss_all_epoch = []
    val_auc_all_epoch = []
    val_prauc_all_epoch = []
    max_prauc_epoch = 0.
    
    ## training and validation loops with balanced folds
    for epoch in range(0, num_epochs):
        
        # generate indices to balance out training set
        _, balance_idx, = create_balanced_bag_subset(y_train_all,
                                                     minority_class_ratio=conf_train.minority_class_ratio,
                                                     rnd_seed=int(epoch*1.5+3*epoch))
        
        # here actually getting the subset needed
        train_dataset = h5_Dataset(conf_preproc["emb_h5"], train_df.iloc[balance_idx])  
        
        # create the pytorch data loader
        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, 
                                                           batch_size=batch_size,
                                                           num_workers=6, shuffle=False)

        model.train()
        train_loss = 0.
        train_error = 0.
        
        # Iterate over data
        for (data, label) in train_dataset_loader:
            data = data.to(device, dtype=torch.float, non_blocking=True)
            label = label.to(device, dtype=torch.long, non_blocking=True)
            
            logits, Y_prob, Y_hat, _, _ = model(data)

            loss = loss_fn(logits, label)
            loss_value = loss.item()

            train_loss += loss_value

            error = calculate_error(Y_hat, label)
            train_error += error

            loss = loss #/ gc
            loss.backward()

            # step
            optimizer.step()
            optimizer.zero_grad()
            
        if conf_train["use_scheduler"]:
            scheduler.step()

        # calculate loss and error for epoch
        train_loss /= len(train_dataset_loader)
        train_error /= len(train_dataset_loader)
        

        ## VALIDATION
        val_dataset = h5_Dataset(conf_preproc["emb_h5"], val_df)

        # create the pytorch data loader
        val_dataset_loader = torch.utils.data.DataLoader(val_dataset, 
                                                         batch_size=128, 
                                                         num_workers=7, shuffle=False)

        stop, val_loss_epoch, auc_epoch, prauc_epoch = validate( model, val_dataset_loader,
                                                                 n_classes, loss_fn, results_dir )
        
        if prauc_epoch > max_prauc_epoch:
            max_prauc_epoch = prauc_epoch
            
        print('EPOCH: %3d    Train loss: %.3f    Val loss: %.3f    ROC_AUC: %.3f    PR_AUC: %.3f    Max PR_AUC: %.4f' % (epoch, train_loss, val_loss_epoch, auc_epoch, prauc_epoch, max_prauc_epoch))       


        os.makedirs(results_dir + f"cv_{cur}/", exist_ok=True)
        torch.save(model.state_dict(), os.path.join(results_dir, f"cv_{cur}", 
                                                    f"trainloss_{np.round(train_loss,4)}_valloss_{np.round(val_loss_epoch,4)}_auc_{np.round(auc_epoch,4)}_prauc_{np.round(prauc_epoch,4)}_"\
                                                    +"checkpoint.pt"))

        val_loss_all_epoch.append(val_loss_epoch)
        val_auc_all_epoch.append(auc_epoch)
        val_prauc_all_epoch.append(prauc_epoch)

        train_loss_all_epoch.append(train_loss)
    
    # Save training parameters to disk    
    param_dict = {'num_epochs': num_epochs,
                  'weight_decay': weight_decay,
                  'train_loss_all_epoch': train_loss_all_epoch,
                  'val_loss_all_epoch': val_loss_all_epoch,
                  'val_auc_all_epoch': val_auc_all_epoch,
                  'val_prauc_all_epoch': val_prauc_all_epoch}
 
    return param_dict
    
def plot_roc(y_true, y_pred):

    #plt.figure(figsize=(4, 4))
    class_ind = 1 #for minority
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, class_ind])
    auc = roc_auc_score(y_true, y_pred[:, class_ind])
    pr_auc = average_precision_score(y_true, y_pred[:, class_ind])
    
    #print('ROC_AUC : %.3f, PR_AUC : %.3f' % (auc, pr_auc))

def validate(model, loader, n_classes, loss_fn = None, results_dir=None):
    
    device=torch.device(f"cuda:{current_gpu}" if torch.cuda.is_available() else "cpu") 
    
    # init variables and set mode to evaluation
    model.eval()
    val_loss = 0.
    val_error = 0.
    
    # containers to store variables to evaluate predictions on VAL set
    prob = [] 
    labels = []

    with torch.no_grad():
        for batch in loader:

            data, label = batch
            data = data.to(device, dtype=torch.float, non_blocking=True)
            label = label.to(device, dtype=torch.long, non_blocking=True)
            
            logits, Y_prob, Y_hat, _, _ = model(data)
            
            loss = loss_fn(logits, label)
            
            # save predictions for current batch
            prob.append(Y_prob.cpu().numpy())
            
            # save labels for current batch
            labels.append(label.cpu().numpy())
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
    val_error /= len(loader)
    val_loss /= len(loader)
    
    prob = np.concatenate(prob)
    labels = np.concatenate(labels)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        prauc = average_precision_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
        
        auc_separated = []
        labels_oh = F.one_hot(torch.from_numpy(labels).to(torch.int64), 5)
        for class_ind in range(prob.shape[-1]):
            fpr, tpr, _ = roc_curve(labels_oh[:, class_ind], prob[:, class_ind])
            auc_current = np.round( roc_auc_score(labels_oh[:, class_ind], prob[:, class_ind]), 3 )
            auc_separated.append(str(auc_current))
    
    # print roc curve
    plot_roc(labels, prob)

    return False, val_loss, auc, prauc
    

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error
    

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device=torch.device(f"cuda:{current_gpu}" if torch.cuda.is_available() else "cpu") 
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



for n_experiment in range(conf_train.n_experiment):

    print('\n')
    print('\n')
    print(f'##################################################################')
    print(f'###################### RUN - {n_experiment} ######################')
    print(f'##################################################################')
    print('\n')
    print('\n')
    max_val_aucs_all_epoch = []

    for i in range(FOLD, FOLD+1):

        print(f'\n ############################ CV-Fold {i} - Balanced training ############################')

        seed_torch(n_experiment)

        print('\nInit loss function...', end=' ')
        loss_fn = nn.CrossEntropyLoss()
        print('Done!')

        print('\nInit Model...', end=' ')
        device=torch.device(f"cuda:{current_gpu}" if torch.cuda.is_available() else "cpu") 
        
        model = ClsTokenTransformerClassifier(conf_train.emb_dim, 
                                      conf_train.num_heads, 
                                      conf_train.num_encoder_layers, 
                                      conf_train.dim_feedforward,
                                      conf_train.dropout,
                                      conf_train.num_classes).to(device)
        print('Done!')
        train_df = pd.read_csv(f'{conf_preproc[splits]}train_split_stratified_{i}.csv')
        val_df = pd.read_csv(f'{conf_preproc[splits]}val_split_stratified_{i}.csv')

        n_classes=2        
        
        results_dir = conf_train.results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        OmegaConf.save(config=OmegaConf.create(conf_train), f=conf_train.results_dir+"conf_train.yaml")

        num_epochs = conf_train.num_epochs
        batch_size = conf_train.batch_size

        param_dict = train_loop( i, train_df.tb_positive.values, val_df.tb_positive.values, 
                                 results_dir, batch_size, num_epochs, model, n_classes, 
                                 loss_fn, gc=32)
        
        max_val_aucs_all_epoch.append( np.max(param_dict['val_auc_all_epoch']) )

        json_data = {'num_epochs': param_dict['num_epochs'],
                     'weight_decay': param_dict['weight_decay'],
                     'train_loss_all_epoch_cv': param_dict['train_loss_all_epoch'],
                     'val_loss_all_epoch_all_cv': param_dict['val_loss_all_epoch'],
                     'val_auc_all_epoch_all_cv': param_dict['val_auc_all_epoch'],
                     'min_val_loss': np.min(param_dict['val_loss_all_epoch']),
                     'max_val_auc': np.max(param_dict['val_auc_all_epoch'])}

        # Save training parameters to disk    
        with open(results_dir + f"cv_{i}/" 'test_params.json', 'w') as file:
            json.dump(json_data, file)
                   
    print('Max AUC in folds: ', max_val_aucs_all_epoch)
    print('Mean AUC: ', np.mean(max_val_aucs_all_epoch))
