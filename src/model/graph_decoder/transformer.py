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
from .layers import Attention, MLP
from .conditions import TimestepEmbedder, ConditionEmbedder, TextEmbedder
from .diffusion_utils import PlaceHolder

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Transformer(nn.Module):
    def __init__(
        self,
        max_n_nodes=50,
        hidden_size=1024,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        drop_condition=0.,
        Xdim=16,
        Edim=5,
        ydim=10,
        text_dim=768,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ydim = ydim
        self.x_embedder = nn.Sequential(
            nn.Linear(Xdim + max_n_nodes * Edim, hidden_size, bias=False),
            nn.LayerNorm(hidden_size)
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ConditionEmbedder(ydim, hidden_size, drop_condition)
        self.txt_embedder = TextEmbedder(text_dim, hidden_size, drop_condition)
        
        self.blocks = nn.ModuleList(
            [
                Block(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.output_layer = OutputLayer(
            max_n_nodes=max_n_nodes,
            hidden_size=hidden_size,
            atom_type=Xdim,
            bond_type=Edim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)

        for block in self.blocks:
            _constant_init(block.adaLN_modulation[0], 0)
        _constant_init(self.output_layer.adaLN_modulation[0], 0)

    def disable_grads(self):
        """
        Disable gradients for all parameters in the model.
        """
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, X_in, E_in, node_mask, y_in, txt, t, unconditioned):
        bs, n, _ = X_in.size()
        X = torch.cat([X_in, E_in.reshape(bs, n, -1)], dim=-1)
        X = self.x_embedder(X)
        
        c1 = self.t_embedder(t)
        c2 = self.y_embedder(y_in, self.training, unconditioned)
        c3 = self.txt_embedder(txt, self.training, unconditioned)
        c = c1 + c2 + c3
        
        for i, block in enumerate(self.blocks):
            X = block(X, c, node_mask)
            
        # X: B * N * dx, E: B * N * N * de
        X, E = self.output_layer(X, X_in, E_in, c, t, node_mask)
        return PlaceHolder(X=X, E=E, y=None).mask(node_mask)

class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=False)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=False, qk_norm=True, **block_kwargs
        )

        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            nn.Softsign()
        )
        
    def forward(self, x, c, node_mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * modulate(self.attn_norm(self.attn(x, node_mask=node_mask)), shift_msa, scale_msa)
        x = x + gate_mlp.unsqueeze(1) * modulate(self.mlp_norm(self.mlp(x)), shift_mlp, scale_mlp)

        return x
    
class OutputLayer(nn.Module):
    def __init__(self, max_n_nodes, hidden_size, atom_type, bond_type, mlp_ratio, num_heads=None):
        super().__init__()
        self.atom_type = atom_type
        self.bond_type = bond_type
        final_size = atom_type + max_n_nodes * bond_type
        self.xedecoder = MLP(in_features=hidden_size, 
                            out_features=final_size, drop=0)

        self.norm_final = nn.LayerNorm(final_size, eps=1e-05, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x, x_in, e_in, c, t, node_mask):
        x_all = self.xedecoder(x)
        B, N, D = x_all.size()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_all = modulate(self.norm_final(x_all), shift, scale)
        
        atom_out = x_all[:, :, :self.atom_type]
        atom_out = x_in + atom_out

        bond_out = x_all[:, :, self.atom_type:].reshape(B, N, N, self.bond_type)
        bond_out = e_in + bond_out

        ##### standardize adj_out
        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = (
            torch.eye(N, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .type_as(edge_mask)
        )
        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        bond_out = 1 / 2 * (bond_out + torch.transpose(bond_out, 1, 2))

        return atom_out, bond_out