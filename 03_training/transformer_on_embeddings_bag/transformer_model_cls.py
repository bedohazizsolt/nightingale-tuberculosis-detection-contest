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


import torch
import torch.nn as nn
import torch.nn.functional as F

class ClsTokenTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, dim_feedforward, dropout, num_classes):
        super(ClsTokenTransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.linear = nn.Linear(input_dim, num_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, N, features)
        batch_size = src.size(1)
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        src = torch.cat((cls_tokens, src), dim=0)
        transformer_output = self.transformer_encoder(src)
        cls_output = transformer_output[0, :, :]  # First token in sequence
        output = self.linear(cls_output)

        return output, F.softmax(output, dim=1), torch.topk(output, 1, dim = 1)[1], None, None
