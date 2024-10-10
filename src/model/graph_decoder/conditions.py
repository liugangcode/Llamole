# Copyright 2024 the Llamole team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
        
class ConditionEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_drop = nn.Embedding(input_size, hidden_size)
        
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size, bias=True),
                nn.Softmax(dim=1),
                nn.Linear(hidden_size, hidden_size, bias=False)
            ) for _ in range(input_size)
        ])

        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, unconditioned):
        embeddings = 0
        for dim in range(labels.shape[1]):
            label = labels[:, dim]
            if unconditioned: 
                drop_ids = torch.ones_like(label).bool()
            else:
                drop_ids = torch.isnan(label)
                if train:
                    random_tensor = torch.rand(label.shape).type_as(labels)
                    probability_mask = random_tensor < self.dropout_prob
                    drop_ids = drop_ids | probability_mask

            label = label.unsqueeze(1)
            embedding = torch.zeros((label.shape[0], self.hidden_size)).type_as(labels)
            mlp_out = self.mlps[dim](label[~drop_ids])
            embedding[~drop_ids] = mlp_out.type_as(embedding)
            embedding[drop_ids] += self.embedding_drop.weight[dim]
            if train:
                embedding = embedding + torch.randn_like(embedding)
            embeddings += embedding

        return embeddings
    
class TextEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_drop = nn.Embedding(1, hidden_size)
        self.linear = nn.Linear(input_size, hidden_size)
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

    def forward(self, text_emb, train, unconditioned):
        if unconditioned: 
            drop_ids = torch.ones(text_emb.shape[0]).bool().to(text_emb.device)
        else:
            drop_ids = torch.isnan(text_emb.sum(dim=1))
            if train:
                random_tensor = torch.rand(text_emb.shape[0]).type_as(text_emb)
                probability_mask = random_tensor < self.dropout_prob
                drop_ids = drop_ids | probability_mask

        embeddings = torch.zeros((text_emb.shape[0], self.hidden_size)).type_as(text_emb)
        linear_out = self.linear(text_emb[~drop_ids])
        embeddings[~drop_ids] = linear_out.type_as(embeddings)
        embeddings[drop_ids] += self.embedding_drop.weight[0]
        
        return embeddings