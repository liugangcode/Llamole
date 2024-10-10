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
import yaml
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import diffusion_utils as utils
from .molecule_utils import graph_to_smiles, check_valid
from .transformer import Transformer

class GraphDiT(nn.Module):
    def __init__(
        self,
        model_config_path,
        data_info_path,
        model_dtype,
    ):
        super().__init__()

        dm_cfg, data_info = utils.load_config(model_config_path, data_info_path)

        input_dims = data_info.input_dims
        output_dims = data_info.output_dims
        nodes_dist = data_info.nodes_dist
        active_index = data_info.active_index

        self.model_config = dm_cfg
        self.data_info = data_info
        self.T = dm_cfg.diffusion_steps
        self.guide_scale = dm_cfg.guide_scale
        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        self.node_dist = nodes_dist
        self.active_index = active_index
        self.max_n_nodes = data_info.max_n_nodes
        self.train_loss = TrainLossDiscrete(dm_cfg.lambda_train)
        self.atom_decoder = data_info.atom_decoder
        self.hidden_size = dm_cfg.hidden_size
        self.text_input_size = 768

        self.denoiser = Transformer(
            max_n_nodes=self.max_n_nodes,
            hidden_size=dm_cfg.hidden_size,
            depth=dm_cfg.depth,
            num_heads=dm_cfg.num_heads,
            mlp_ratio=dm_cfg.mlp_ratio,
            drop_condition=dm_cfg.drop_condition,
            Xdim=self.Xdim,
            Edim=self.Edim,
            ydim=self.ydim,
            text_dim=self.text_input_size
        )
        self.model_dtype = model_dtype

        self.noise_schedule = utils.PredefinedNoiseScheduleDiscrete(
            dm_cfg.diffusion_noise_schedule, timesteps=dm_cfg.diffusion_steps
        )
        x_marginals = data_info.node_types.to(self.model_dtype) / torch.sum(
            data_info.node_types.to(self.model_dtype)
        )
        e_marginals = data_info.edge_types.to(self.model_dtype) / torch.sum(
            data_info.edge_types.to(self.model_dtype)
        )
        x_marginals = x_marginals / x_marginals.sum()
        e_marginals = e_marginals / e_marginals.sum()

        xe_conditions = data_info.transition_E.to(self.model_dtype)
        xe_conditions = xe_conditions[self.active_index][:, self.active_index]

        xe_conditions = xe_conditions.sum(dim=1)
        ex_conditions = xe_conditions.t()
        xe_conditions = xe_conditions / xe_conditions.sum(dim=-1, keepdim=True)
        ex_conditions = ex_conditions / ex_conditions.sum(dim=-1, keepdim=True)

        self.transition_model = utils.MarginalTransition(
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            xe_conditions=xe_conditions,
            ex_conditions=ex_conditions,
            y_classes=self.ydim_output,
            n_nodes=self.max_n_nodes,
        )
        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals, y=None)

    def init_model(self, model_dir, verbose=False):
        model_file = os.path.join(model_dir, 'model.pt')
        if os.path.exists(model_file):
            self.denoiser.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        if verbose:
            print('GraphDiT Denoiser Model initialized.')
            print('Denoiser model:\n', self.denoiser)

    def save_pretrained(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save model
        model_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.denoiser.state_dict(), model_path)
        
        # Save model config
        config_path = os.path.join(output_dir, 'model_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(vars(self.model_config), f)
        
        # Save data info
        data_info_path = os.path.join(output_dir, 'data.meta.json')
        data_info_dict = {
            "active_atoms": self.data_info.active_atoms,
            "max_node": self.data_info.max_n_nodes,
            "n_atoms_per_mol_dist": self.data_info.n_nodes.tolist(),
            "bond_type_dist": self.data_info.edge_types.tolist(),
            "transition_E": self.data_info.transition_E.tolist(),
            "atom_type_dist": self.data_info.node_types.tolist(),
            "valencies": self.data_info.valency_distribution.tolist()
        }
        with open(data_info_path, 'w') as f:
            json.dump(data_info_dict, f, indent=2)
        
        print('GraphDiT Model and configurations saved to:', output_dir)

    def disable_grads(self):
        self.denoiser.disable_grads()
    
    def forward(
        self, x, edge_index, edge_attr, graph_batch, properties, text_embedding, no_label_index
    ):
        properties = torch.where(properties == no_label_index, float("nan"), properties)
        data_x = F.one_hot(x, num_classes=118).to(self.model_dtype)[
            :, self.active_index
        ]
        data_edge_attr = F.one_hot(edge_attr, num_classes=5).to(self.model_dtype)

        dense_data, node_mask = utils.to_dense(
            data_x, edge_index, data_edge_attr, graph_batch, self.max_n_nodes
        )
        X, E = dense_data.X, dense_data.E

        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(X, E, properties, node_mask)
        pred = self._forward(noisy_data, text_embedding)
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            node_mask=node_mask,
        )
        return loss

    def _forward(self, noisy_data, text_embedding, unconditioned=False):
        noisy_x, noisy_e, properties = (
            noisy_data["X_t"].to(self.model_dtype),
            noisy_data["E_t"].to(self.model_dtype),
            noisy_data["y_t"].to(self.model_dtype).clone(),
        )
        node_mask, timestep, text_embedding = (
            noisy_data["node_mask"],
            noisy_data["t"],
            text_embedding.to(self.model_dtype),
        )
        
        pred = self.denoiser(
            noisy_x,
            noisy_e,
            node_mask,
            properties,
            text_embedding,
            timestep,
            unconditioned=unconditioned,
        )
        return pred

    def apply_noise(self, X, E, y, node_mask):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device
        ).to(
            self.model_dtype
        )  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(
            alpha_t_bar, X.device
        )  # (bs, dx_in, dx_out), (bs, de_in, de_out)

        bs, n, d = X.shape
        X_all = torch.cat([X, E.reshape(bs, n, -1)], dim=-1)
        prob_all = X_all @ Qtb.X
        probX = prob_all[:, :, : self.Xdim_output]
        probE = prob_all[:, :, self.Xdim_output :].reshape(bs, n, n, -1)

        sampled_t = utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        y_t = y
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y_t).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }
        return noisy_data

    @torch.no_grad()
    def generate(
        self,
        properties,
        text_embedding,
        no_label_index,
    ):
        properties = torch.where(properties == no_label_index, float("nan"), properties)
        batch_size = properties.size(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_nodes = self.node_dist.sample_n(batch_size, device)
        arange = (
            torch.arange(self.max_n_nodes, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        z_T = utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E = z_T.X, z_T.E

        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        y = properties
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, text_embedding, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
            
        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        smiles_list = graph_to_smiles(molecule_list, self.atom_decoder)

        return smiles_list

    def check_valid(self, smiles):
        return check_valid(smiles)
    
    def sample_p_zs_given_zt(
        self, s, t, X_t, E_t, properties, text_embedding, node_mask
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, _ = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": properties,
            "t": t,
            "node_mask": node_mask,
        }

        def get_prob(noisy_data, text_embedding, unconditioned=False):
            pred = self._forward(noisy_data, text_embedding, unconditioned=unconditioned)

            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            device = text_embedding.device
            # Retrieve transitions matrix
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device)
            Qt = self.transition_model.get_Qt(beta_t, device)

            Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
            predX_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)

            unnormalized_probX_all = utils.reverse_diffusion(
                predX_0=predX_all, X_t=Xt_all, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )

            unnormalized_prob_X = unnormalized_probX_all[:, :, : self.Xdim_output]
            unnormalized_prob_E = unnormalized_probX_all[
                :, :, self.Xdim_output :
            ].reshape(bs, n * n, -1)

            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5

            prob_X = unnormalized_prob_X / torch.sum(
                unnormalized_prob_X, dim=-1, keepdim=True
            )  # bs, n, d_t-1
            prob_E = unnormalized_prob_E / torch.sum(
                unnormalized_prob_E, dim=-1, keepdim=True
            )  # bs, n, d_t-1
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

            return prob_X, prob_E

        prob_X, prob_E = get_prob(noisy_data, text_embedding)

        ### Guidance
        if self.guide_scale is not None and self.guide_scale != 1:
            uncon_prob_X, uncon_prob_E = get_prob(
                noisy_data, text_embedding, unconditioned=True
            )
            prob_X = (
                uncon_prob_X
                * (prob_X / uncon_prob_X.clamp_min(1e-5)) ** self.guide_scale
            )
            prob_E = (
                uncon_prob_E
                * (prob_E / uncon_prob_E.clamp_min(1e-5)) ** self.guide_scale
            )
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        sampled_s = utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask, step=s[0, 0].item()
        )

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).to(self.model_dtype)
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).to(self.model_dtype)

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=properties)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=properties)

        return out_one_hot.mask(node_mask).type_as(properties), out_discrete.mask(
            node_mask, collapse=True
        ).type_as(properties)
    

class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy"""

    def __init__(self, lambda_train):
        super().__init__()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, node_mask):

        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(
            masked_pred_X, (-1, masked_pred_X.size(-1))
        )  # (bs * n, dx)
        masked_pred_E = torch.reshape(
            masked_pred_E, (-1, masked_pred_E.size(-1))
        )  # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.0).any(dim=-1)
        mask_E = (true_E != 0.0).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        target_X = torch.argmax(flat_true_X, dim=-1)
        loss_X = F.cross_entropy(flat_pred_X, target_X, reduction="mean")

        target_E = torch.argmax(flat_true_E, dim=-1)
        loss_E = F.cross_entropy(flat_pred_E, target_E, reduction="mean")

        total_loss = self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_E

        return total_loss
