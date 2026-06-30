"""
Critic networks.

"""

from typing import Union
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + identity)

class QEnsemble(nn.Module):
    """Ensemble of Q networks with optional target networks."""

    def __init__(self, network,
                 num_ensemble=2, use_target=False,
                 init_std=0.01, target_update_mom=0.995):
        super().__init__()
        self.num_ensemble = num_ensemble
        self.init_std = init_std
        self.target_update_mom = target_update_mom
        self.q_nets = nn.ModuleList([copy.deepcopy(network) for _ in range(num_ensemble)])
        for q_net in self.q_nets:
            self._init_module(q_net)

        self.use_target = use_target
        if use_target:
            self.target_q_nets = nn.ModuleList([copy.deepcopy(network) for _ in range(num_ensemble)])
            for target_param, param in zip(self.target_q_nets.parameters(), self.q_nets.parameters()):
                target_param.data.copy_(param.data)
                target_param.requires_grad = False

    def _init_module(self, network):
        for m in network.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

    def update_target_networks(self):
        if self.use_target:
            for target_net, net in zip(self.target_q_nets, self.q_nets):
                for target_param, param in zip(target_net.parameters(), net.parameters()):
                    target_param.data.copy_(self.target_update_mom * target_param.data + (1 - self.target_update_mom) * param.data)
        else:
            logging.warning("Target networks not used in this QEnsemble.")

    def compute_target(self, states, actions, reduction="min", _q_net=None):
        assert states.shape[0] == actions.shape[0], "Batch size of states and actions must match."
        assert states.ndim == actions.ndim, "States and actions must have the same number of dimensions."
        state_dim, action_dim = states.shape, actions.shape

        q_values = []
        if _q_net is None:
            _q_net = self.target_q_nets if self.use_target else self.q_nets
        for target_net in _q_net:
            q_val = target_net({"state": states.view(-1, state_dim[-1])}, actions.view(-1, action_dim[-1]))
            q_values.append(q_val)

        if len(self.q_nets) > 1:
            q_values = torch.stack(q_values, dim=0)  # (num_ensemble, batch_size, 1)
            if reduction == "min":
                return_q, _ = torch.min(q_values, dim=0)
            elif reduction == "mean":
                return_q = torch.mean(q_values, dim=0)
            else:
                raise NotImplementedError(f"Reduction {reduction} not implemented.")
            q_std = torch.std(q_values, dim=0)
        else:
            return_q = q_values[0]
            q_std = torch.zeros_like(return_q)

        return_q = return_q.reshape(state_dim[:-1])
        q_std = q_std.reshape(state_dim[:-1])
        return return_q, q_std

    def forward(self, states, actions, reduction="min"):
        return self.compute_target(states, actions, reduction=reduction, _q_net=self.q_nets)

    def compute_gradq(self, states, actions, qnet_idx=-1):
        assert states.shape[0] == actions.shape[0], "Batch size of states and actions must match."
        assert states.ndim == actions.ndim, "States and actions must have the same number of dimensions."
        state_dim, action_dim = states.shape, actions.shape

        gradq_values = []
        _q_net = self.q_nets
        with torch.enable_grad():
            for target_net in _q_net:
                _actions = actions.clone().detach().requires_grad_(True)
                q_val = target_net({"state": states.view(-1, state_dim[-1])}, _actions.view(-1, action_dim[-1]))
                grad_q = torch.autograd.grad(
                    q_val.sum(), _actions, create_graph=False
                )[0]
                grad_q = grad_q.reshape(action_dim)
                gradq_values.append(grad_q.detach())

        if len(self.q_nets) > 1 and qnet_idx < 0:
            gradq_values = torch.stack(gradq_values, dim=0)  # (num_ensemble, B, P, A)
            gradq_mean = torch.mean(gradq_values, dim=0)  # (B, P, A)
        elif qnet_idx >= 0:
            gradq_mean = gradq_values[qnet_idx]
        else:
            gradq_mean = gradq_values[0]

        return gradq_mean

    def compute_loss(self, states, actions, q_targets, split="traj", mask=None):
        state_dim, action_dim = states.shape, actions.shape

        total_loss = 0.
        q_error, q_vals = [], []

        if mask is None:
            mask = torch.ones_like(q_targets, dtype=torch.bool)

        states_flatten, action_flatten, q_targets_flatten, mask_flatten = \
            states.reshape(-1, state_dim[-1]), actions.reshape(-1, action_dim[-1]), q_targets.reshape(-1), mask.reshape(-1)

        if split == "chunk":
            state_chunks    = torch.chunk(states_flatten, self.num_ensemble, dim=0)
            action_chunks   = torch.chunk(action_flatten, self.num_ensemble, dim=0)
            q_targets_chunks = torch.chunk(q_targets_flatten, self.num_ensemble, dim=0)
            mask_chunks     = torch.chunk(mask_flatten, self.num_ensemble, dim=0)
        elif split == "traj":
            state_chunks    = [states[idx::self.num_ensemble].reshape(-1, state_dim[-1])   for idx in range(self.num_ensemble)]
            action_chunks   = [actions[idx::self.num_ensemble].reshape(-1, action_dim[-1]) for idx in range(self.num_ensemble)]
            q_targets_chunks = [q_targets[idx::self.num_ensemble].reshape(-1)              for idx in range(self.num_ensemble)]
            mask_chunks     = [mask[idx::self.num_ensemble].reshape(-1)                    for idx in range(self.num_ensemble)]
        elif split == "state":
            state_chunks    = [states_flatten[idx::self.num_ensemble].reshape(-1, state_dim[-1])   for idx in range(self.num_ensemble)]
            action_chunks   = [action_flatten[idx::self.num_ensemble].reshape(-1, action_dim[-1])  for idx in range(self.num_ensemble)]
            q_targets_chunks = [q_targets_flatten[idx::self.num_ensemble].reshape(-1)              for idx in range(self.num_ensemble)]
            mask_chunks     = [mask_flatten[idx::self.num_ensemble].reshape(-1)                    for idx in range(self.num_ensemble)]

        for idx, target_net in enumerate(self.q_nets):
            q_val_partition = target_net({"state": state_chunks[idx]}, action_chunks[idx])
            mask_chunk = mask_chunks[idx]

            critic_loss = torch.mean(mask_chunk * (q_val_partition.view(-1) - q_targets_chunks[idx]) ** 2)
            total_loss += critic_loss

            q_error.append(torch.sqrt(critic_loss).item())
            q_vals.append(q_val_partition.mean().item())

        return total_loss, np.mean(q_error), np.mean(q_vals)

    def compute_mean_var(self, states, actions, temp=1.):
        """Compute Q-weighted mean and Cholesky covariance over action particles.

        Args:
            states:  (N, P, S)
            actions: (N, P, A)
        Returns:
            mean: (N, A), L: (N, A, A)
        """
        Qval, Qstd = self.forward(states, actions)

        maxQ = torch.max(Qval, dim=1)[0]
        weight = torch.exp((Qval - maxQ.unsqueeze(-1)) / temp).unsqueeze(-1)  # (N, P, 1)
        weight = weight / torch.sum(weight, dim=1, keepdim=True)

        mean = torch.sum(actions * weight, dim=1)  # (N, A)
        cov = torch.sum(
            weight.unsqueeze(-1)
            * (actions[:, :, :, None] - mean[:, None, :, None])
            * (actions[:, :, None, :] - mean[:, None, None, :]),
            dim=1
        )

        bias = torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0) * 1e-6
        L = torch.linalg.cholesky(cov + bias)
        return mean, L
    

def _flat_cond(obs, obs_dims):
    """ Flatten to (N, 1, *)  """
    return {k: obs[k].reshape(-1, 1, *obs_dims[k]) for k in obs_dims}

class QImgEnsemble(nn.Module):
    """Ensemble of Q networks with visual encoders and optional target networks."""
    def __init__(self, network, obs_dims,
                num_ensemble=2, use_target=False, 
                init_std=0.01, target_update_mom=0.995):
        super().__init__()
        self.obs_dims = obs_dims
        self.num_ensemble = num_ensemble
        self.init_std = init_std
        self.target_update_mom = target_update_mom
        
        self.q_nets = nn.ModuleList([copy.deepcopy(network.Q1) for _ in range(num_ensemble)])
        del network.Q1
        self.feat_encoder = network
        for q_net in self.q_nets:
            self._init_module(q_net)

        self.use_target = use_target
        if use_target:
            self.target_encoder = copy.deepcopy(self.feat_encoder)
            self.target_q_nets = nn.ModuleList([copy.deepcopy(self.q_nets[0]) for _ in range(num_ensemble)])
            for target_param, param in zip(self.target_q_nets.parameters(), self.q_nets.parameters()):
                target_param.data.copy_(param.data)
                target_param.requires_grad = False
            
            for param in self.target_encoder.parameters():
                param.requires_grad = False
    
    def forward_q(self, Qnet, action: torch.Tensor, feat: torch.Tensor):
        """
        Compute Q value(s) from a precomputed feature and action.
        feat:   (B, feat_dim)
        action: (B, Ta, Da)
        """
        B = feat.shape[0]
        action = action.view(B, -1)
        x = torch.cat((feat, action), dim=-1)
        return Qnet(x).squeeze(1)

    def _init_module(self, network):
        for m in network.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

    def update_target_networks(self):
        if self.use_target:
            for target_net, net in zip(self.target_q_nets, self.q_nets):
                for target_param, param in zip(target_net.parameters(), net.parameters()):
                    target_param.data.copy_(self.target_update_mom * target_param.data + (1 - self.target_update_mom) * param.data)
        
            for target_param, param in zip(self.target_encoder.parameters(), self.feat_encoder.parameters()):
                target_param.data.copy_(self.target_update_mom * target_param.data + (1 - self.target_update_mom) * param.data)
        else:
            logging.warning("Target networks not used in this QEnsemble.")

    def compute_target(self, cond, actions, reduction="min", compute_target=True):
        """
        Compute the target Q
        
        Args:
            cond: dict in shape (T is the history)
                'state' - (N, T, C)
                'rgb'   - (N, T, C, H, W)
            actions: (N, A)
            _q_net is None if target, else target
        """
        action_dim = actions.shape
        state_dim_temporal = cond['state'].shape[:2]
    
        # for each ensemble member, compute Q values
        q_values = []
        _q_net = self.target_q_nets if compute_target else self.q_nets
        _encoder = self.target_encoder if compute_target else self.feat_encoder
        
        encode_feat = _encoder.encode_feature(cond)
        for target_net in _q_net:
            # target_net(cond, actions.view(-1, action_dim[-1]))
            q_val = self.forward_q(target_net, actions.view(-1, action_dim[-1]), encode_feat) 
            q_values.append(q_val)

        # reduce to single output
        if len(self.q_nets) > 1:
            q_values = torch.stack(q_values, dim=0)  # (num_ensemble, flat_batch, 1)
            if reduction == "min":
                return_q, _ = torch.min(q_values, dim=0)
            elif reduction == "mean":
                return_q = torch.mean(q_values, dim=0)
            else:
                raise NotImplementedError(f"Reduction {reduction} not implemented.")
            q_std = torch.std(q_values, dim=0)
        else:
            return_q = q_values[0]
            q_std = torch.zeros_like(return_q)
        
        return_q = return_q.reshape(state_dim_temporal)
        return return_q, q_std

    def forward(self, cond, actions, reduction="min"):
        """ Always use current Q networks for forward pass """
        return self.compute_target(cond, actions, reduction=reduction, compute_target=False)

    @torch.no_grad()
    def compute_gradq(self, cond, actions, qnet_idx=-1):
        """ Compute the Q gradient
        
        Args:
            cond (dict) - dict for cond input
            actions (N, C)
            qnet_idx (int) - which q net gradient is used as output.
        """
        action_dim = actions.shape
        batch_leading_ndim = actions.ndim - 1
        # flat_cond, _ = self._make_flat_cond(cond, batch_leading_ndim)

        # for each ensemble member, compute Q values
        gradq_values = []
        _q_net = self.q_nets
        
        # precompute feature; it does not need gradients.
        encode_feat = self.feat_encoder.encode_feature(_flat_cond(cond, self.obs_dims), no_augment=True)
        
        with torch.enable_grad():
            for target_net in _q_net:
                _actions = actions.clone().detach().requires_grad_(True)
                q_val = self.forward_q(target_net, _actions.view(-1, action_dim[-1]), encode_feat) 
                grad_q = torch.autograd.grad(
                    q_val.sum(), _actions, create_graph=False
                )[0]
                grad_q = grad_q.reshape(action_dim)
                gradq_values.append(grad_q.detach())

        if len(self.q_nets) > 1 and qnet_idx < 0:    
            gradq_values = torch.stack(gradq_values, dim=0)  # (num_ensemble, B, P, A)
            gradq_mean = torch.mean(gradq_values, dim=0)  # (B, P, A)
        elif qnet_idx >= 0:
            gradq_mean = gradq_values[qnet_idx]
        else:
            gradq_mean = gradq_values[0]

        return gradq_mean

    def compute_loss(self, cond, actions, q_targets, mask=None):
        """
        Compute Loss for the TD-Lambda Learning

        Args:   
        Args:
            cond: dict in shape (T is the timstep)
                'state' - (N, T, C)
                'rgb'   - (N, T, C, H, W)
            actions: (N, T, A)
            q_targets - (N, T)
        """ 
        action_dim = actions.shape

        # for each ensemble member, compute Q values
        total_loss = 0.
        q_error = []
        q_vals = []

        if mask is None:
            mask = torch.ones_like(q_targets, dtype=torch.bool)

        # trajectory chunk
        cond_chunks, action_chunks, q_targets_chunks, mask_chunks  = [], [], [], []
        for idx in range(self.num_ensemble):
            cond_chunks.append({k:cond[k][idx::self.num_ensemble] for k in self.obs_dims} )
            action_chunks.append(actions[idx::self.num_ensemble].reshape(-1, action_dim[-1]))
            q_targets_chunks.append( q_targets[idx::self.num_ensemble].reshape(-1)) 
            mask_chunks.append(mask[idx::self.num_ensemble].reshape(-1))

        encode_feats = [self.feat_encoder.encode_feature(_flat_cond(cond, self.obs_dims)) for cond in cond_chunks]
        for idx, target_net in enumerate(self.q_nets):
            q_val_partition = self.forward_q(target_net, action_chunks[idx], encode_feats[idx]) 
            # q_val_partition = target_net(_flat_cond(cond_chunks[idx], self.obs_dims), action_chunks[idx])
            mask_chunk = mask_chunks[idx]

            critic_loss = torch.mean(mask_chunk * (q_val_partition.view(-1) - q_targets_chunks[idx]) ** 2)
            total_loss += critic_loss
            q_error.append(torch.sqrt(critic_loss).item())
            q_vals.append(q_val_partition.mean().item())

        return total_loss, np.mean(q_error), np.mean(q_vals)
    
    def compute_mean_var(self, states, actions, temp=1.):
        """ Compute the mean and variance for each env based on Q 
        
        Args:
            states: (N, P, C)
            actions: (N, P, C)
            
        Return 
            mean: (N, C)
            L: (N, C, C) - cholesky decomp
        """
        Qval, Qstd = self.forward(states, actions)
        
        maxQ = torch.max(Qval, dim=1)[0]
        weight = torch.exp( (Qval - maxQ.unsqueeze(-1)) / temp ).unsqueeze(-1) # (B, P, 1)
        weight = weight / torch.sum(weight, dim = 1, keepdim=True)
        
        # compute weighted mean
        mean = torch.sum( actions * weight, dim = 1) # (B, C)
        cov = torch.sum( 
                weight.unsqueeze(-1) * (actions[:, :, :, None] - mean[:, None, :, None]) * (actions[:, :, None, :] - mean[:, None, None, :]) , dim = 1
            )
        
        bias = torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0) * 1e-6
        L = torch.linalg.cholesky(cov + bias) # (B, C, C)
        return mean, L

class CriticObs(torch.nn.Module):
    """State-only critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim] + mlp_dims + [1]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """
        if isinstance(cond, dict):
            B = len(cond["state"])

            # flatten history
            state = cond["state"].view(B, -1)
        else:
            state = cond
        q1 = self.Q1(state)
        return q1


class CriticObsAct(torch.nn.Module):
    """State-action double critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        double_q=True,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.double_q = double_q
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if double_q:
            self.Q2 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
            

    def compute_q(self, action: torch.Tensor, feat: torch.Tensor):
        """
        Compute Q value(s) from a precomputed feature and action.
        feat:   (B, feat_dim)
        action: (B, Ta, Da)
        """
        B = feat.shape[0]
        action = action.view(B, -1)
        x = torch.cat((feat, action), dim=-1)
        if hasattr(self, "Q2"):
            return self.Q1(x).squeeze(1), self.Q2(x).squeeze(1)
        else:
            return self.Q1(x).squeeze(1)

    def forward(self, cond: dict, action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        B = len(cond["state"])

        # flatten history
        state = cond["state"].view(B, -1)

        # flatten action
        action = action.view(B, -1)

        x = torch.cat((state, action), dim=-1)
        if hasattr(self, "Q2"):
            q1 = self.Q1(x)
            q2 = self.Q2(x)
            return q1.squeeze(1), q2.squeeze(1)
        else:
            q1 = self.Q1(x)
            return q1.squeeze(1)


class ViTCriticObsAct(CriticObsAct):
    """ViT + MLP, state-action double critic."""

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        # update input dim to mlp (visual features + state will be concatenated before action)
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def forward(
        self,
        cond: dict,
        action: torch.Tensor,
        no_augment=False,
    ):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        action: (B, Ta, Da)
        no_augment: whether to skip augmentation
        """
        feat = self.encode_feature(cond, no_augment=no_augment)
        return self.compute_q(action, feat)

    def encode_feature(self, cond: dict, no_augment=False):
        """
        Encode visual and state observations into a feature vector.
        cond:
            state: (B, To, Do)
            rgb:   (B, To, C, H, W)
        Returns feat: (B, spatial_emb * num_img + state_dim)
        """
        B, T_rgb, C, H, W = cond["rgb"].shape
        state = cond["state"].view(B, -1)
        rgb = cond["rgb"][:, -self.img_cond_steps:]

        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        rgb = rgb.float()

        if self.num_img > 1:
            rgb1, rgb2 = rgb[:, 0], rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.compress1.forward(self.backbone(rgb1), state)
            feat2 = self.compress2.forward(self.backbone(rgb2), state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:
            if self.augment and not no_augment:
                rgb = self.aug(rgb)
            feat = self.compress.forward(self.backbone(rgb), state)
        return torch.cat([feat, state], dim=-1)


class CNNCriticObsAct(CriticObsAct):
    """CNN + MLP, state-action double critic."""

    def __init__(
        self,
        cond_dim,
        cnn_feature_dim=256,
        img_cond_steps=1,
        augment=False,
        num_img=1,
        num_channel=3,
        img_h=96,
        img_w=96,
        **kwargs,
    ):
        # MLP input: CNN features (per image) + encoded state (128)
        state_emb_dim = 128
        mlp_obs_dim = cnn_feature_dim * num_img + state_emb_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        self.augment = augment

        self.state_encoder = nn.Sequential(
            nn.Linear(cond_dim, state_emb_dim // 2),
            nn.LayerNorm(state_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(state_emb_dim // 2, state_emb_dim),
        )
        in_channels = num_channel * img_cond_steps
        self.cnn = self._build_cnn(in_channels, cnn_feature_dim, img_h, img_w)
        
        if augment:
            self.aug = RandomShiftsAug(pad=4)

    @staticmethod
    def _cnn_output_size(img_h, img_w):
        """Compute flattened size after all five conv layers."""
        def conv_out(size, kernel, stride):
            return (size - kernel) // stride + 1
        h = conv_out(img_h, 8, 4)   # → ~23 for 96
        h = conv_out(h, 4, 2)        # → ~10
        h = conv_out(h, 3, 1)        # → ~8
        h = conv_out(h, 2, 2)        # → 4
        h = conv_out(h, 2, 2)        # → 2
        w = conv_out(img_w, 8, 4)
        w = conv_out(w, 4, 2)
        w = conv_out(w, 3, 1)
        w = conv_out(w, 2, 2)
        w = conv_out(w, 2, 2)
        return 128 * h * w

    def _build_cnn(self, in_channels, feature_dim, img_h, img_w):
        flat_dim = self._cnn_output_size(img_h, img_w)
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),   # → 4×4
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=2),  # → 2×2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

    def encode_feature(self, cond: dict, no_augment=False):
        """
        Encode visual and state observations into a feature vector.
        cond:
            state: (B, To, Do)
            rgb:   (B, To, C, H, W)
        Returns feat: (B, cnn_feature_dim * num_img + 128)
        """
        B, T_rgb, C, H, W = cond["rgb"].shape
        state = self.state_encoder(cond["state"].view(B, -1))
        rgb = cond["rgb"][:, -self.img_cond_steps:]

        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        
        # normalize the image if in [0-255]
        if rgb.max() > 1:
            rgb = rgb.float() / 255.0

        if self.num_img > 1:
            rgb1, rgb2 = rgb[:, 0], rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.cnn(rgb1)
            feat2 = self.cnn(rgb2)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:
            if self.augment and not no_augment:
                rgb = self.aug(rgb)
            feat = self.cnn(rgb)
        return torch.cat([feat, state], dim=-1)

    def forward(
        self,
        cond: dict,
        action: torch.Tensor,
        no_augment=False,
    ):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        action: (B, Ta, Da)
        """
        feat = self.encode_feature(cond, no_augment=no_augment)
        return self.compute_q(action, feat)


class ResNetCriticObsAct(CriticObsAct):
    """ResNet CNN backbone + SpatialEmb + MLP, state-action double critic.

    Replaces CNNCriticObsAct's global-average-pool + independent state_encoder
    with SpatialEmb, so proprioception conditions the visual aggregation before
    the Q MLP. This makes spatial feature selection state-aware.

    Backbone produces (B, 256, h_out, w_out) features derived from img_h/img_w,
    reshaped to (B, h_out*w_out, 256) patches for SpatialEmb — no adaptive
    pooling needed since num_patch is computed exactly from the input resolution.
    """

    @staticmethod
    def _spatial_output_size(img_h, img_w):
        """Compute (h_out, w_out) of the backbone for a given input resolution."""
        def conv_out(size, kernel, stride):
            return (size - kernel) // stride + 1
        h = conv_out(img_h, 8, 4)
        h = conv_out(h, 4, 2)
        h = conv_out(h, 3, 1)
        h = conv_out(h, 2, 2)
        h = conv_out(h, 2, 2)
        w = conv_out(img_w, 8, 4)
        w = conv_out(w, 4, 2)
        w = conv_out(w, 3, 1)
        w = conv_out(w, 2, 2)
        w = conv_out(w, 2, 2)
        return h, w

    def __init__(
        self,
        cond_dim,
        spatial_emb=128,
        cnn_feature_dim=256,
        img_cond_steps=1,
        dropout=0,
        augment=False,
        num_img=1,
        num_channel=3,
        img_h=96,
        img_w=96,
        **kwargs,
    ):
        h_out, w_out = self._spatial_output_size(img_h, img_w)
        num_patch = h_out * w_out
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)

        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        self.augment = augment

        in_channels = num_channel * img_cond_steps
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),     # /4   96→23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),               # /2   23→10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),               # /1   10→8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),              # /2    8→4
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.Conv2d(128, cnn_feature_dim, kernel_size=2, stride=2), # /2    4→2
            nn.GroupNorm(4, cnn_feature_dim),
            nn.ReLU(),
        )
        self.cnn_feature_dim = cnn_feature_dim

        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=num_patch,
                patch_dim=cnn_feature_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:
            self.compress = SpatialEmb(
                num_patch=num_patch,
                patch_dim=cnn_feature_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )

        if augment:
            self.aug = RandomShiftsAug(pad=4)

    def _compress_img(self, img, state, compress):
        x = self.backbone(img)                               # (B, 256, h_out, w_out)
        B = x.shape[0]
        x = x.view(B, self.cnn_feature_dim, -1).transpose(1, 2)  # (B, num_patch, 256)
        return compress(x, state)                            # (B, spatial_emb)

    def encode_feature(self, cond: dict, no_augment=False):
        B, T_rgb, C, H, W = cond["rgb"].shape
        state = cond["state"].view(B, -1)
        rgb = cond["rgb"][:, -self.img_cond_steps:]

        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        if rgb.max() > 1:
            rgb = rgb.float() / 255.0

        if self.num_img > 1:
            rgb1, rgb2 = rgb[:, 0], rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self._compress_img(rgb1, state, self.compress1)
            feat2 = self._compress_img(rgb2, state, self.compress2)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:
            if self.augment and not no_augment:
                rgb = self.aug(rgb)
            feat = self._compress_img(rgb, state, self.compress)

        return torch.cat([feat, state], dim=-1)


class ViTCritic(CriticObs):
    """ViT + MLP, state only"""

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        # update input dim to mlp
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:  # TODO: clean up
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def forward(
        self,
        cond: dict,
        no_augment=False,
    ):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        no_augment: whether to skip augmentation

        TODO long term: more flexible handling of cond
        """
        B, T_rgb, C, H, W = cond["rgb"].shape

        # flatten history
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment and not no_augment:
                rgb = self.aug(rgb)  # uint8 -> float32
            feat = self.backbone(rgb)
            feat = self.compress.forward(feat, state)
        feat = torch.cat([feat, state], dim=-1)
        return super().forward(feat)
