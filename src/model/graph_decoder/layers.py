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

from torch.jit import Final
import torch.nn.functional as F
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn

class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0,
        proj_drop=0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )  # FIXME
        assert self.fast_attn, "scaled_dot_product_attention Not implemented"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, node_mask):
        B, N, D = x.shape

        # B, head, N, head_dim
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # B, head, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = (node_mask[:, None, :, None] & node_mask[:, None, None, :]).expand(
            -1, self.num_heads, N, N
        )
        extended_nodes = (attn_mask.sum(dim=-1) == 0)
        attn_mask = attn_mask.clone()
        attn_mask[extended_nodes] = True

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p,
            attn_mask=attn_mask,
        )

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)
