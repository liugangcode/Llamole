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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.nn import MessagePassing
import json

class GraphCLIP(nn.Module):
    def __init__(
        self,
        graph_num_layer,
        graph_hidden_size,
        dropout,
        model_config,
    ):
        super().__init__()
        self.model_config = model_config
        self.hidden_size = graph_hidden_size
        self.molecule_encoder = GNNEncoder(num_layer=graph_num_layer, hidden_size=graph_hidden_size, drop_ratio=dropout)
        self.molecule_projection = ProjectionHead(embedding_dim=graph_hidden_size, projection_dim=graph_hidden_size, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        molecule_features = self.molecule_encoder(x, edge_index, edge_attr, batch)
        molecule_embeddings = self.molecule_projection(molecule_features)
        molecule_embeddings = molecule_embeddings / molecule_embeddings.norm(dim=-1, keepdim=True)
        return molecule_embeddings
    
    def save_pretrained(self, output_dir):
        """
        Save the molecule encoder, projection models, and model_config to the output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        molecule_path = os.path.join(output_dir, 'model.pt')
        proj_path = molecule_path.replace('model', 'model_proj')
        config_path = os.path.join(output_dir, 'model_config.json')

        torch.save(self.molecule_encoder.state_dict(), molecule_path)
        torch.save(self.molecule_projection.state_dict(), proj_path)
        
        # Save model_config to JSON file
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f, indent=2)

    def disable_grads(self):
        """
        Disable gradients for all parameters in the model.
        """
        for param in self.parameters():
            param.requires_grad = False

    def init_model(self, model_path, verbose=True):
        molecule_path = os.path.join(model_path, 'model.pt')
        proj_path = molecule_path.replace('model', 'model_proj')
        if os.path.exists(molecule_path):
            self.molecule_encoder.load_state_dict(torch.load(molecule_path, map_location='cpu', weights_only=False))
        else:
            raise FileNotFoundError(f"Molecule encoder file not found: {molecule_path}")
        
        if os.path.exists(proj_path):
            self.molecule_projection.load_state_dict(torch.load(proj_path, map_location='cpu', weights_only=False))
        else:
            raise FileNotFoundError(f"Molecule projection file not found: {proj_path}")
        
        if verbose:
            print('GraphCLIP Models initialized.')
            print('Molecule model:\n', self.molecule_encoder)
            print('Molecule projection:\n', self.molecule_projection)

class GNNEncoder(nn.Module):
    def __init__(self, num_layer, hidden_size, drop_ratio):

        super(GNNEncoder, self).__init__()
        
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = nn.Embedding(118, hidden_size)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = nn.Embedding(1, hidden_size)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        
        ### List of GNNs
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mlp_virtualnode_list = nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GINConv(hidden_size, drop_ratio))
            self.norms.append(nn.LayerNorm(hidden_size, elementwise_affine=True))
            if layer < num_layer - 1:
                self.mlp_virtualnode_list.append(nn.Sequential(nn.Linear(hidden_size, 4*hidden_size), nn.LayerNorm(4*hidden_size), nn.GELU(), nn.Dropout(drop_ratio), \
                                                               nn.Linear(4*hidden_size, hidden_size)))
            
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x, edge_index, edge_attr, batch):

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]   
        
            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.norms[layer](h)
            
            if layer < self.num_layer - 1:
                h = F.gelu(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)
            
            h = h + h_list[layer]
            h_list.append(h)

            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtual_pool = global_max_pool(h_list[layer], batch)
                virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtual_pool), self.drop_ratio, training = self.training)

        h_node = h_list[-1]
        h_graph = global_add_pool(h_node, batch)
        
        return h_graph
    
class GINConv(MessagePassing):
    def __init__(self, hidden_size, drop_ratio):
        '''
            hidden_size (int)
        '''
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = nn.Sequential(nn.Linear(hidden_size, 4*hidden_size), nn.LayerNorm(4*hidden_size), nn.GELU(), nn.Dropout(drop_ratio), nn.Linear(4*hidden_size, hidden_size))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = nn.Embedding(5, hidden_size)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.gelu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
        act_layer=nn.GELU,
        hidden_features=None,
        bias=True
    ):
        super().__init__()
        projection_dim = projection_dim or embedding_dim
        hidden_features = hidden_features or embedding_dim
        linear_layer = nn.Linear

        self.fc1 = linear_layer(embedding_dim, hidden_features, bias=bias)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = linear_layer(hidden_features, projection_dim, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x
