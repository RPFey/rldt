import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Union, List
from torch.distributions import Normal
from agent.finetune.train_qfm_img_agent import QImgAgent
import numpy as np
import logging
import math
import einops
import glob
import copy
import time as time_m
import wandb
import pickle
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
from util.timer import Timer
from util.reward_scaling import EMARewardScaler
from model.common.modules import RandomShiftsAug
from util.file_io import HDF5Fetcher
from model.common.critic import QImgEnsemble, _flat_cond
Sample = namedtuple("Sample", "trajectories chains")

class FMSVGDImgAgent(QImgAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # SVGD configuration
        self.svgd_type = cfg.train.svgd_type
        self.svgd_temp = cfg.train.svgd_temp
        self.rbf_bandwith = cfg.train.rbf_bandwith
        # SVGD Los weight
        self.consistency_coef = cfg.train.consistency_coef
        self.kl_coef = cfg.train.kl_coef
        
        self.n_particles = cfg.train.n_particles
        self.q_norm = cfg.train.q_norm
        self.svgd_type = cfg.train.svgd_type
        self.sqrt_distance = cfg.train.sqrt_distance
        self.q_norm_max, self.q_norm_min, self.q_norm_reg = cfg.train.q_norm_max, cfg.train.q_norm_min, cfg.train.q_norm_reg

    @torch.no_grad()
    def batch_svgd_direction(self, query, actions, states, q_net, final_ind=None):
        """
        Get the SVGD directions on the sampled actions.
        Args:
            query: (B, P1, A) - expected actions from intermediate states
            actions: (B, P2, A) - sampled actions
            states:  (B, P2, S)
            final_ind: (B, P1) whether the query is the final step. 1 - final step; 0 - intermediate step
        """
        B, P1, _ = query.shape
        B, P2, _ = actions.shape

        # compute the pairwise distance between action samples, to monitor the diversity of the samples
        action_sample_distance = torch.cdist(actions, actions).mean()

        grad_q = q_net.compute_gradq(
            states, actions, self.gradq_net_idx
        )  # (B, P2, A)
        
        grad_q_norm = torch.norm(grad_q, dim=-1, keepdim=True)  # (B, P2, 1)
        avg_grad_q_norm = torch.mean(grad_q_norm).item()
        
        if self.svgd_type == 'delta':
            return grad_q, {"q_norm": avg_grad_q_norm, "grad_K_norm": 0.0, "action_sample_distance": action_sample_distance.item()}
        
        avg_grad_q_norm_batch = torch.mean(grad_q_norm, dim=1)  # (B, 1)
        avg_grad_q_norm_batch.clamp_(max=self.q_norm_max, min=self.q_norm_min) # (B, 1)

        svgd_temp = self.svgd_temp
        if self.q_norm == 'q_grad':
            grad_q = grad_q / (avg_grad_q_norm_batch.view(-1, 1, 1) + self.q_norm_reg)  # normalize the gradient to have unit norm, to avoid the scale issue between grad_q and grad_K
        elif self.q_norm == 'svgd':
            svgd_temp = self.svgd_temp * avg_grad_q_norm_batch.view(-1, 1, 1) # scale up the repulsion term when the gradient is large, to encourage more exploration
        
        # Compute the RBF kernel and its gradients
        # grad_q = grad_q.detach().view(B, P, -1) # (B, P2, A)
        distances = torch.cdist(query, actions) ** 2  # (B, P1, P2), squre distance

        # take the median as the bandwidth
        if self.rbf_bandwith <= 0:
            median_dist = distances.median(dim=-1).values  # (B, P1)
            if self.sqrt_distance:
                rbf_bandwith = torch.sqrt(median_dist) / ( 2 * math.log(P2 + 1) )  # (B, P1)
            else:
                rbf_bandwith = median_dist / ( 2 * math.log(P2 + 1) )  # (B, P1)
            rbf_bandwith = rbf_bandwith.view(B, P1, 1)  # (B, P1, 1)
        else:
            rbf_bandwith = query.new_full((B, P1, 1), float(self.rbf_bandwith))

        # avoid division instability for tiny bandwidth values
        rbf_bandwith = torch.clamp(rbf_bandwith, min=1e-8)
        inv_bandwidth = 1.0 / rbf_bandwith
        
        # compute Kernel
        K = torch.exp(-0.5 * distances * inv_bandwidth)  # (B, P1, P2)

        # Attraction term: sum_j K_ij * grad_q_j
        attract = torch.einsum("bij,bjk->bik", K, grad_q)  # (B, P1, A)

        # Repulsion term: sum_j grad_x K_ij, without materializing (B, P1, P2, A)
        # grad_x K_ij = (x_i - a_j) / h_i * K_ij
        sum_k = K.sum(dim=2, keepdim=True)  # (B, P1, 1)
        weighted_actions = torch.einsum("bij,bjk->bik", K, actions)  # (B, P1, A)
        repel = (query * sum_k - weighted_actions) * inv_bandwidth  # (B, P1, A)

        # Equivalent norm metric for grad_K, but computed from distances to avoid huge 4D tensor.
        avg_grad_K_norm = torch.mean(distances * K * inv_bandwidth).item()
        
        if final_ind is not None:
            # keep grad_K same for terminal sample ( repulsion between terminal samples)
            # flip for non-terminal samples ( consistency )
            final_ind = 2 * final_ind - 1 # 1 - final step; -1 - intermediate step
            repel = repel * final_ind.view(B, P1, 1)
        
        phi = (attract + svgd_temp * repel) / P2  # (B, P1, A)
        return phi, {"q_norm": avg_grad_q_norm, "grad_K_norm": avg_grad_K_norm, "action_sample_distance": action_sample_distance.item()}
    
    @torch.no_grad()
    def update_flow_with_svgd(self, batch, loss_scale=1.0):
        """
        One policy update step.
        """
        # Build full cond dict (handles both state-only and image inputs)
        cond = {k: batch[k].to(self.device) for k in self.obs_dims if k in batch}
        batch_size = cond["state"].shape[0]
        ode_steps = self.model.ode_steps

        # ---------------------------------
        # 1. Compute SVGD direction
        # ---------------------------------
        # repeat each obs key along batch dim for n_particles
        cond_sample = {k: v.repeat_interleave(self.n_particles, dim=0) for k, v in cond.items()}
        _action_sample, cond_encoded = self.model.cache_sampling(_flat_cond(cond, self.obs_dims), 
                        num_replica=self.n_particles, deterministic=False, forward_net=self.model.actor_ft)
        action_sample = _action_sample.trajectories

        # ---------------------------------
        # 2. Flow matching update
        # ---------------------------------
        # compute the expected targets for point on trajectory
        B, _, _ = action_sample.shape
        ode_step = torch.randint(0, ode_steps, (B,), device=cond["state"].device)
        final_ind = (ode_step == ode_steps - 1).float() if self.svgd_type == "hybrid" else None
        t = ode_step.float() / ode_steps
        update_chain = _action_sample.chains[torch.arange(B, device=cond["state"].device), ode_step]  # (B, horizon, act)

        model_pred = self.model.actor_ft.predict_velocity(
            cond_encoded.repeat_interleave(self.n_particles, dim=0), 
            update_chain, t)  # (B * P, act), predict the velocity
        
        if self.model.prediction_type == "v_pred":
            exp_land_point = update_chain + (1 - t).view(B, 1, 1) * model_pred  # compute the expected landing point
        elif self.model.prediction_type == "x_pred":
            exp_land_point = model_pred

        # reshape obs to 3D (batch, particles, *obs_shape) for SVGD direction
        cond_sample_3d = {k: v.view(batch_size, self.n_particles, *v.shape[1:]) for k, v in cond_sample.items()}
        action_sample_3d = action_sample[:, :self.act_steps, :].view(batch_size, self.n_particles, -1)
        exp_land_point_3d = exp_land_point[:, :self.act_steps, :].view(batch_size, self.n_particles, -1)

        # compute SVGD
        if self.svgd_type == 'delta':
            phi, svgd_info = self.batch_svgd_direction(
                exp_land_point_3d,
                exp_land_point_3d,
                cond_sample_3d,
                self.q_net,
                final_ind = None)  # (M, P, C)
        else:
            phi, svgd_info = self.batch_svgd_direction(
                exp_land_point_3d,
                action_sample_3d,
                cond_sample_3d,
                self.q_net,
                final_ind = final_ind)  # (M, P, C)
        phi = phi.reshape(batch_size * self.n_particles, self.act_steps, -1)  # (M * P, C)
        
        # Fix problems when act step < horizon step
        if self.act_steps < self.horizon_steps:
            phi = torch.cat(
                [phi, 
                 torch.zeros( 
                    (batch_size * self.n_particles, self.horizon_steps - self.act_steps, self.action_dim) , device=phi.device)]
            )

        # compute the teacher output on old samples
        old_t = torch.randint(0, self.model.ode_steps, (B,), device=cond["state"].device).float() / self.model.ode_steps
        old_t_view = old_t.view(B, 1, 1)
        old_x_t = (1 - old_t_view) * torch.randn_like(action_sample) + old_t_view * action_sample
        
        # get old target from self.model.actor using cond_encoded and predict_velocity
        old_encoded_feat = self.model.actor.encode_rgb(_flat_cond(cond, self.obs_dims)) # !! Use old feature encoder
        old_target = self.model.actor.predict_velocity(
            old_encoded_feat.repeat_interleave(self.n_particles, dim=0),
            old_x_t, old_t)

        # update the model
        with torch.enable_grad():
            # Get the encoded feature fist
            encoded_feat = self.model.actor_ft.encode_rgb(_flat_cond(cond, self.obs_dims))

            # repeat the feature and get the model prediction
            updated_model_pred = self.model.actor_ft.predict_velocity(
                encoded_feat.repeat_interleave(self.n_particles, dim=0), update_chain, t)

            # compute SVGD
            dot_prod = torch.sum(updated_model_pred * phi.view_as(action_sample), dim=(-1, -2))
            dot_prod = dot_prod / (ode_steps * (1 - t)) if self.model.prediction_type == "v_pred" else dot_prod / ode_steps  # (B * P,)
            svgd_ft_loss = -1 * torch.mean(dot_prod)

            # consistency loss
            if self.model.prediction_type == "v_pred":
                exp_land_point_ = update_chain + (1 - t).view(B, 1, 1) * updated_model_pred  # compute the expected landing point
                consistency_loss = self.consistency_coef * torch.mean(
                    torch.sum((exp_land_point_ - action_sample) ** 2, dim=(-1, -2)) / ((ode_steps * (1 - t)) ** 2)
                )
            elif self.model.prediction_type == "x_pred":
                exp_land_point_ = updated_model_pred
                consistency_loss = self.consistency_coef * torch.mean(
                    torch.sum((exp_land_point_ - action_sample) ** 2, dim=(-1, -2)))

            # KL loss; get the prediction on old x and t
            old_flow_prediction = self.model.actor_ft.predict_velocity(
                encoded_feat.repeat_interleave(self.n_particles, dim=0), old_x_t, old_t)
            kl_loss = torch.mean(torch.sum((old_flow_prediction - old_target) ** 2, dim=(-1, -2))) * self.kl_coef

            ft_loss = svgd_ft_loss + kl_loss + consistency_loss
            (ft_loss * loss_scale).backward()

        loss_info = {
            "actor_loss": svgd_ft_loss.item(),
            "kl_loss": kl_loss.item(),
            "consistency_loss": consistency_loss.item(),
        }
        loss_info.update(svgd_info)
        return loss_info
    
    def _update_actor(self, step_idx, batch_t, loss_scale=1.0):
        """ Update actor with QAM adjoint matching loss. Caller owns zero_grad / optimizer.step / lr_scheduler.step. """
        return self.update_flow_with_svgd(batch_t, loss_scale)
