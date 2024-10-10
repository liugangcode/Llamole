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
import numpy as np
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops
import os
import json
import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    return SimpleNamespace(
        **{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )

class DataInfos:
    def __init__(self, meta_filename="data.meta.json"):
        self.all_targets = ["BBBP", "HIV", "BACE", "CO2", "N2", "O2", "FFV", "TC"]
        self.task_type = "pretrain"
        if os.path.exists(meta_filename):
            with open(meta_filename, "r") as f:
                meta_dict = json.load(f)
        else:
            raise FileNotFoundError(f"Meta file {meta_filename} not found.")

        self.active_atoms = meta_dict["active_atoms"]
        self.max_n_nodes = meta_dict["max_node"]
        self.original_max_n_nodes = meta_dict["max_node"]
        self.n_nodes = torch.Tensor(meta_dict["n_atoms_per_mol_dist"])
        self.edge_types = torch.Tensor(meta_dict["bond_type_dist"])
        self.transition_E = torch.Tensor(meta_dict["transition_E"])

        self.atom_decoder = meta_dict["active_atoms"]
        node_types = torch.Tensor(meta_dict["atom_type_dist"])
        active_index = (node_types > 0).nonzero().squeeze()
        self.node_types = torch.Tensor(meta_dict["atom_type_dist"])[active_index]
        self.nodes_dist = DistributionNodes(self.n_nodes)
        self.active_index = active_index

        val_len = 3 * self.original_max_n_nodes - 2
        meta_val = torch.Tensor(meta_dict["valencies"])
        self.valency_distribution = torch.zeros(val_len)
        val_len = min(val_len, len(meta_val))
        self.valency_distribution[:val_len] = meta_val[:val_len]
        self.input_dims = {"X": 16, "E": 5, "y": 10}
        self.output_dims = {"X": 16, "E": 5, "y": 10}


def load_config(config_path, data_meta_info_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not os.path.exists(data_meta_info_path):
        raise FileNotFoundError(f"Data meta info file not found: {data_meta_info_path}")

    with open(config_path, "r") as file:
        cfg_dict = yaml.safe_load(file)

    cfg = dict_to_namespace(cfg_dict)

    data_info = DataInfos(data_meta_info_path)
    return cfg, data_info


### graph utils
class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor, categorical: bool = False):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if categorical:
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def to_dense(x, edge_index, edge_attr, batch, max_num_nodes=None):
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    # node_mask = node_mask.float()
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if max_num_nodes is None:
        max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


### diffusion utils
class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p

class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        betas = cosine_beta_schedule_discrete(timesteps)
        self.register_buffer("betas", torch.from_numpy(betas).float())

        # 0.9999
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=1)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        self.betas = self.betas.type_as(t_int)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        self.alphas_bar = self.alphas_bar.type_as(t_int)
        return self.alphas_bar[t_int.long()]


class DiscreteUniformTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device, X=None, flatten_e=None):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(
            self.X_classes, device=device
        ).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(
            self.E_classes, device=device
        ).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(
            self.y_classes, device=device
        ).unsqueeze(0)

        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device, X=None, flatten_e=None):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = (
            alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_x
        )
        q_e = (
            alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_e
        )
        q_y = (
            alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_y
        )

        return PlaceHolder(X=q_x, E=q_e, y=q_y)


class MarginalTransition:
    def __init__(
        self, x_marginals, e_marginals, xe_conditions, ex_conditions, y_classes, n_nodes
    ):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals  # Dx
        self.e_marginals = e_marginals  # Dx, De
        self.xe_conditions = xe_conditions
        # print('e_marginals.dtype', e_marginals.dtype)
        # print('x_marginals.dtype', x_marginals.dtype)
        # print('xe_conditions.dtype', xe_conditions.dtype)

        self.u_x = (
            x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        )  # 1, Dx, Dx
        self.u_e = (
            e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        )  # 1, De, De
        self.u_xe = xe_conditions.unsqueeze(0)  # 1, Dx, De
        self.u_ex = ex_conditions.unsqueeze(0)  # 1, De, Dx
        self.u = self.get_union_transition(
            self.u_x, self.u_e, self.u_xe, self.u_ex, n_nodes
        )  # 1, Dx + n*De, Dx + n*De

    def get_union_transition(self, u_x, u_e, u_xe, u_ex, n_nodes):
        u_e = u_e.repeat(1, n_nodes, n_nodes)  # (1, n*de, n*de)
        u_xe = u_xe.repeat(1, 1, n_nodes)  # (1, dx, n*de)
        u_ex = u_ex.repeat(1, n_nodes, 1)  # (1, n*de, dx)
        u0 = torch.cat([u_x, u_xe], dim=2)  # (1, dx, dx + n*de)
        u1 = torch.cat([u_ex, u_e], dim=2)  # (1, n*de, dx + n*de)
        u = torch.cat([u0, u1], dim=1)  # (1, dx + n*de, dx + n*de)
        return u

    def index_edge_margin(self, X, q_e, n_bond=5):
        # q_e: (bs, dx, de) --> (bs, n, de)
        bs, n, n_atom = X.shape
        node_indices = X.argmax(-1)  # (bs, n)
        ind = node_indices[:, :, None].expand(bs, n, n_bond)
        q_e = torch.gather(q_e, 1, ind)
        return q_e

    def get_Qt(self, beta_t, device):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K
        beta_t: (bs)
        returns: q (bs, d0, d0)
        """
        bs = beta_t.size(0)
        d0 = self.u.size(-1)
        self.u = self.u.to(device)
        u = self.u.expand(bs, d0, d0)

        beta_t = beta_t.to(device)
        beta_t = beta_t.view(bs, 1, 1)
        q = beta_t * u + (1 - beta_t) * torch.eye(d0, device=device, dtype=self.u.dtype).unsqueeze(0)

        return PlaceHolder(X=q, E=None, y=None)

    def get_Qt_bar(self, alpha_bar_t, device):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K
        alpha_bar_t: (bs, 1) roduct of the (1 - beta_t) for each time step from 0 to t.
        returns: q (bs, d0, d0)
        """
        bs = alpha_bar_t.size(0)
        d0 = self.u.size(-1)
        alpha_bar_t = alpha_bar_t.to(device)
        alpha_bar_t = alpha_bar_t.view(bs, 1, 1)
        self.u = self.u.to(device)
        q = (
            alpha_bar_t * torch.eye(d0, device=device, dtype=self.u.dtype).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u
        )

        return PlaceHolder(X=q, E=None, y=None)


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]
    
def assert_correctly_masked(variable, node_mask):
    assert (
        variable * (1 - node_mask.long())
    ).abs().max().item() < 1e-4, "Variables not masked properly."


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def sample_discrete_features(probX, probE, node_mask, step=None, add_nose=True):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _ = probX.shape

    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    probX = probX.clamp_min(1e-5)
    probX = probX / probX.sum(dim=-1, keepdim=True)
    X_t = probX.multinomial(1)  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)
    probE = probE.clamp_min(1e-5)
    probE = probE / probE.sum(dim=-1, keepdim=True)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    # Add a small value everywhere to avoid nans
    pred_X = pred_X.clamp_min(1e-5)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)

    pred_E = pred_E.clamp_min(1e-5)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = torch.ones(true_X.size(-1), dtype=true_X.dtype, device=true_X.device)
    row_E = torch.zeros(
        true_E.size(-1), dtype=true_E.dtype, device=true_E.device
    ).clamp_min(1e-5)
    row_E[0] = 1.0

    diag_mask = ~torch.eye(
        node_mask.size(1), device=node_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    true_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_X[~node_mask] = row_X.type_as(pred_X)
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = (
        row_E.type_as(pred_E)
    )

    return true_X, true_E, pred_X, pred_E


def forward_diffusion(X, X_t, Qt, Qsb, Qtb, X_dim):
    bs, n, d = X.shape

    Qt_X_T = torch.transpose(Qt.X, -2, -1)  # (bs, d, d)
    left_term = X_t @ Qt_X_T  # (bs, N, d)
    right_term = X @ Qsb.X  # (bs, N, d)

    numerator = left_term * right_term  # (bs, N, d)
    denominator = X @ Qtb.X  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denominator = denominator * X_t

    num_X = numerator[:, :, :X_dim]
    num_E = numerator[:, :, X_dim:].reshape(bs, n * n, -1)

    deno_X = denominator[:, :, :X_dim]
    deno_E = denominator[:, :, X_dim:].reshape(bs, n * n, -1)

    denominator = denominator.unsqueeze(-1)  # (bs, N, 1)

    deno_X = deno_X.sum(dim=-1, keepdim=True)
    deno_E = deno_E.sum(dim=-1, keepdim=True)

    deno_X[deno_X == 0.0] = 1
    deno_E[deno_E == 0.0] = 1
    prob_X = num_X / deno_X
    prob_E = num_E / deno_E

    prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True)
    prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True)
    return PlaceHolder(X=prob_X, E=prob_E, y=None)


def reverse_diffusion(predX_0, X_t, Qt, Qsb, Qtb):
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
    X_t: bs, n, dt          or bs, n, n, dt
    Qt: bs, d_t-1, dt
    Qsb: bs, d0, d_t-1
    Qtb: bs, d0, dt.
    """
    Qt_T = Qt.transpose(-1, -2)  # bs, N, dt
    assert Qt.dim() == 3
    left_term = X_t @ Qt_T  # bs, N, d_t-1
    right_term = predX_0 @ Qsb
    numerator = left_term * right_term  # bs, N, d_t-1

    denominator = Qtb @ X_t.transpose(-1, -2)  # bs, d0, N
    denominator = denominator.transpose(-1, -2)  # bs, N, d0
    return numerator / denominator.clamp_min(1e-5)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_X = F.one_hot(U_X.long(), num_classes=x_limit.shape[-1]).type_as(x_limit)

    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_E = F.one_hot(U_E.long(), num_classes=e_limit.shape[-1]).type_as(x_limit)

    U_X = U_X.to(node_mask.device)
    U_E = U_E.to(node_mask.device)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()
    return PlaceHolder(X=U_X, E=U_E, y=None).mask(node_mask)


def index_QE(X, q_e, n_bond=5):
    bs, n, n_atom = X.shape
    node_indices = X.argmax(-1)  # (bs, n)

    exp_ind1 = node_indices[:, :, None, None, None].expand(
        bs, n, n_atom, n_bond, n_bond
    )
    exp_ind2 = node_indices[:, :, None, None, None].expand(bs, n, n, n_bond, n_bond)

    q_e = torch.gather(q_e, 1, exp_ind1)
    q_e = torch.gather(q_e, 2, exp_ind2)  # (bs, n, n, n_bond, n_bond)

    node_mask = X.sum(-1) != 0
    no_edge = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
    q_e[no_edge] = torch.tensor([1, 0, 0, 0, 0]).type_as(q_e)

    return q_e
