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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 384, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        
        return A, x
    
    

class HIPT_LGP_FC_STAGE3ONLY(nn.Module):

    def __init__(self, emb_dim, nhead, num_layers, dropout_fc, dropout_transformer, dropout_attn, n_classes=2):
        super(HIPT_LGP_FC_STAGE3ONLY, self).__init__()
        size = 192
        self.global_phi = nn.Sequential(nn.Linear(emb_dim, size), nn.ReLU(), nn.Dropout(dropout_fc))
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=size, nhead=nhead, dim_feedforward=size, dropout=dropout_transformer, activation='relu'
            ), 
            num_layers=num_layers
        )
        self.global_attn_pool = Attn_Net_Gated(L=size, D=size, dropout=dropout_attn, n_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size, size), nn.ReLU(), nn.Dropout(dropout_fc)])
        self.classifier = nn.Linear(size, n_classes)
        
        

    def forward(self, h_4096, **kwargs):
        h_4096 = h_4096.squeeze(0)
        h_4096 = self.global_phi(h_4096)
        h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
        A_4096, h_4096 = self.global_attn_pool(h_4096) 
        A_4096 = torch.transpose(A_4096, 1, 0)
        A_4096 = F.softmax(A_4096, dim=1) 
        h_path = torch.mm(A_4096, h_4096)
        h_WSI = self.global_rho(h_path)
        logits = self.classifier(h_WSI)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        return logits, F.softmax(logits, dim=1), Y_hat, None, None
