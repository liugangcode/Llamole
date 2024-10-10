# Copyright 2024 Llamole Team
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
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing

import os
import json
from collections import defaultdict
from rdchiral.main import rdchiralRunText
import pandas as pd

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class GraphPredictor(nn.Module):
    def __init__(
        self,
        num_layer,
        hidden_size,
        drop_ratio,
        out_dim,
        model_config,
        label_to_template,
        available=None,
    ):
        super().__init__()
        self.model_config = model_config
        self.text_input_size = model_config.get("text_input_size", 768)
        self.available = available
        self.text_drop = drop_ratio
        
        # Process label_to_template
        if isinstance(label_to_template, pd.DataFrame):
            self.label_to_template = dict(
                zip(
                    label_to_template["rule_label"],
                    label_to_template["retro_templates"],
                )
            )
        else:
            self.label_to_template = label_to_template

        self.predictor = GNNRetrosynthsizer(
            num_layer, hidden_size, self.text_input_size, drop_ratio, out_dim
        )
        self.neural_cost = None

    def save_pretrained(self, output_dir):
        """
        Save the predictor model, model_config, label_to_template, and available to the output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_path = os.path.join(output_dir, "model.pt")
        config_path = os.path.join(output_dir, "model_config.json")
        label_to_template_path = os.path.join(output_dir, "label_to_template.csv.gz")
        available_path = os.path.join(output_dir, "available.csv.gz")

        # Save predictor model
        torch.save(self.predictor.state_dict(), model_path)

        # Save cost model
        if self.neural_cost is not None:
            neural_cost_path = os.path.join(output_dir, "cost_model.pt")
            torch.save(self.neural_cost.state_dict(), neural_cost_path)

        # Save model_config to JSON file
        with open(config_path, "w") as f:
            json.dump(self.model_config, f, indent=2)

        # Save label_to_template to gzipped CSV file
        label_to_template_df = pd.DataFrame(
            list(self.label_to_template.items()),
            columns=["rule_label", "retro_templates"],
        )
        label_to_template_df.to_csv(
            label_to_template_path, index=False, compression="gzip"
        )

        # Save available to gzipped CSV file if it's not None
        if self.available is not None:
            if isinstance(self.available, list):
                available_df = pd.DataFrame(self.available, columns=["smiles"])
            elif isinstance(self.available, pd.DataFrame):
                available_df = self.available
            else:
                raise ValueError(
                    "available must be either a list of SMILES strings or a pandas DataFrame"
                )

            available_df.to_csv(available_path, index=False, compression="gzip")

    def disable_grads(self):
        """
        Disable gradients for all parameters in the model.
        """
        for param in self.predictor.parameters():
            param.requires_grad = False

    def init_neural_cost(self, model_path, verbose=False):
        model_file = os.path.join(model_path, "cost_model.pt")
        if os.path.exists(model_file):
            self.neural_cost = CostMLP(
                n_layers=1, fp_dim=2048, latent_dim=128, dropout_rate=0.1
            )
            self.neural_cost.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")

        for param in self.neural_cost.parameters():
            param.requires_grad = False

        if verbose:
            print("Neural Cost Model initialized.")
            print("Neural Cost Model:\n", self.neural_cost)

    def init_model(self, model_path, verbose=False):
        model_file = os.path.join(model_path, "model.pt")
        if os.path.exists(model_file):
            self.predictor.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")

        if verbose:
            print("GraphPredictor Model initialized.")
            print("Predictor model:\n", self.predictor)

    def forward(self, x, edge_index, edge_attr, batch, c):
        return self.predictor(x, edge_index, edge_attr, batch, c)

    def estimate_cost(self, smiles):
        if self.neural_cost is None:
            raise ValueError("Cost model is not initialized.")

        fp = self.neural_cost.smiles_to_fp(smiles)
        dtype, device = (
            next(self.neural_cost.parameters()).dtype,
            next(self.neural_cost.parameters()).device,
        )
        fp = torch.tensor(fp, dtype=dtype, device=device).view(1, -1)
        return self.neural_cost(fp).squeeze().item()


    def sample_templates(self, product_graph, c, product_smiles, topk=10):

        x, edge_index, edge_attr = (
            product_graph.x,
            product_graph.edge_index,
            product_graph.edge_attr,
        )
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Sample from main predictor
        logits_main = self.predictor(x, edge_index, edge_attr, batch, c)
        logits_drop = self.predictor(x, edge_index, edge_attr, batch, None)
        probs_main = logits_main + logits_drop * self.text_drop
        probs_main = F.softmax(logits_main, dim=1)

        topk_probs, topk_indices = torch.topk(probs_main, k=topk, dim=1)

        # Convert to numpy for easier handling
        topk_probs = topk_probs.float().cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

        # Get the corresponding templates
        templates = []
        for idx in topk_indices[0]:
            templates.append(self.label_to_template[idx])
        
        reactants_d = defaultdict(list)
        for prob, template in zip(topk_probs[0], templates):
            try:
                outcomes = rdchiralRunText(template, product_smiles)
                if len(outcomes) == 0:
                    continue
                outcomes = sorted(outcomes)
                for reactant in outcomes:
                    if "." in reactant:
                        str_list = sorted(reactant.strip().split("."))
                        reactants_d[".".join(str_list)].append(
                            (prob.item() / len(outcomes), template)
                        )
                    else:
                        reactants_d[reactant].append(
                            (prob.item() / len(outcomes), template)
                        )
            except Exception:
                pass
        
        if len(reactants_d) == 0:
            return [], [], []

        def merge(reactant_d):
            ret = []
            for reactant, l in reactant_d.items():
                ss, ts = zip(*l)
                ret.append((reactant, sum(ss), list(ts)[0]))
            reactants, scores, templates = zip(
                *sorted(ret, key=lambda item: item[1], reverse=True)
            )
            return list(reactants), list(scores), list(templates)

        reactants, scores, templates = merge(reactants_d)

        total = sum(scores)
        scores = [s / total for s in scores]

        return reactants, scores, templates

class GNNRetrosynthsizer(torch.nn.Module):
    def __init__(self, num_layer, hidden_size, text_input_size, drop_ratio, out_dim):
        super(GNNRetrosynthsizer, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.text_input_size = text_input_size
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = nn.Embedding(118, hidden_size)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = nn.Embedding(1, hidden_size)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.mlp_virtualnode_list = nn.ModuleList()
        
        self.text_dropping = nn.Embedding(1, text_input_size)
        for layer in range(num_layer):
            self.convs.append(GINConv(hidden_size, drop_ratio))
            self.adapters.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.text_input_size, 3 * hidden_size, bias=True),
                )
            )
            self.norms.append(nn.LayerNorm(hidden_size, elementwise_affine=False))
            if layer < num_layer - 1:
                self.mlp_virtualnode_list.append(
                    nn.Sequential(
                        nn.Linear(hidden_size, 4 * hidden_size),
                        nn.LayerNorm(4 * hidden_size),
                        nn.GELU(),
                        nn.Dropout(drop_ratio),
                        nn.Linear(4 * hidden_size, hidden_size),
                    )
                )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.LayerNorm(4 * hidden_size),
            nn.GELU(),
            nn.Dropout(drop_ratio),
            nn.Linear(4 * hidden_size, out_dim),
        )

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

        for adapter in self.adapters:
            _constant_init(adapter[-1], 0)

    def disable_grads(self):
        """
        Disable gradients for all parameters in the model.
        """
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, edge_index, edge_attr, batch, c):

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        h_list = [self.atom_encoder(x)]

        if c is None:
            c = self.text_dropping.weight.expand(batch.max().item() + 1, -1)

        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            shift, scale, gate = self.adapters[layer](c).chunk(3, dim=1)
            # B = batch.max().item() + 1
            node_counts = torch.bincount(batch, minlength=batch.max().item() + 1)
            shift = shift.repeat_interleave(node_counts, dim=0)
            scale = scale.repeat_interleave(node_counts, dim=0)
            gate = gate.repeat_interleave(node_counts, dim=0)

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            # h = self.norms[layer](h)
            h = modulate(self.norms[layer](h), shift, scale)

            if layer < self.num_layer - 1:
                h = F.gelu(h)
                h = F.dropout(h, self.drop_ratio, training=self.training)

            h = gate * h + h_list[layer]
            h_list.append(h)

            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtual_pool = global_max_pool(h_list[layer], batch)
                virtualnode_embedding = virtualnode_embedding + F.dropout(
                    self.mlp_virtualnode_list[layer](virtual_pool),
                    self.drop_ratio,
                    training=self.training,
                )

        h_node = h_list[-1]
        h_graph = global_add_pool(h_node, batch)
        output = self.decoder(h_graph)
        return output


class CostMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate):
        super(CostMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))
        self.layers = nn.Sequential(*layers)

    def smiles_to_fp(self, smiles: str, fp_dim: int = 2048) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
        onbits = list(fp.GetOnBits())
        arr = np.zeros(fp.GetNumBits(), dtype=bool)
        arr[onbits] = 1

        return arr

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))
        return x


class GINConv(MessagePassing):
    def __init__(self, hidden_size, drop_ratio):
        """
        hidden_size (int)
        """
        super(GINConv, self).__init__(aggr="add")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.LayerNorm(4 * hidden_size),
            nn.GELU(),
            nn.Dropout(drop_ratio),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = nn.Embedding(5, hidden_size)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        return out

    def message(self, x_j, edge_attr):
        return F.gelu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
