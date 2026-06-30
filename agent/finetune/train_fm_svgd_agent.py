import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Union, List
from torch.distributions import Normal
from agent.finetune.train_agent import TrainAgent
from model.common.critic import QEnsemble
import numpy as np
import logging
import math
import einops
import glob
import copy
import wandb
import pickle
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
from util.timer import Timer
from util.reward_scaling import EMARewardScaler
Sample = namedtuple("Sample", "trajectories chains")

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class FMSVGDAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
    
        # initialize the optimizer
        self.fm_policy = self.model.actor_ft  # Point to the base flow matching policy
        self.fm_optimizer = torch.optim.Adam(self.fm_policy.parameters(), lr=cfg.train.fm_update_lr, weight_decay=cfg.train.critic_weight_decay)
        self.fm_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.fm_optimizer,
            start_factor=1.0 / cfg.train.fm_lr_scheduler.warmup_steps,
            end_factor=1.0,
            total_iters=cfg.train.fm_lr_scheduler.warmup_steps,
        )

        if hasattr(self.model.critic, "Q2"):
            use_target = True
            del self.model.critic.Q2  # only use single Q for FM-SVGD
        else:
            use_target = False
        self.q_net:QEnsemble = QEnsemble(self.model.critic, cfg.train.ensemble_q, use_target, cfg.train.q_init_std, cfg.train.q_target_mom)
        self.q_optimizer = torch.optim.AdamW(self.q_net.q_nets.parameters(), lr=cfg.train.q_update_lr, weight_decay=cfg.train.critic_weight_decay)
        # Replay buffer for (s, a, r, done) in trajectory
        self.buffer_num_episode = cfg.train.buffer_num_episode * self.n_envs
        # buffer to store episode stats
        self.rollout_stage_cond = cfg.train.rollout_stage_cond
        self.old_stats = dict(
            states=np.empty((0, self.n_steps + 1, self.cfg.obs_dim)), # terminal obs
            actions=np.empty((0, self.n_steps + 1, self.cfg.action_dim * self.cfg.act_steps)),
            rewards=np.empty((0, self.n_steps)),
            done=np.empty((0, self.n_steps)),
            timesteps=np.empty((0, self.n_steps + 1)),
        )
        
        self.gamma = cfg.train.gamma
        self.n_particles = cfg.train.n_particles
        self.q_update_steps = cfg.train.q_update_steps
        self.q_split_mode = cfg.train.q_split_mode
        self.weighted_sample = cfg.train.weighted_sample
        self.policy_update_steps = cfg.train.policy_update_steps
        self.gradq_net_idx = cfg.train.gradq_net_idx
        self.q_bootstrap = cfg.train.q_bootstrap
        self.td_lambda = cfg.train.td_lambda
        self.sqrt_distance = cfg.train.sqrt_distance

        # SVGD configuration
        self.svgd_type = cfg.train.svgd_type
        self.svgd_temp = cfg.train.svgd_temp
        self.rbf_bandwith = cfg.train.rbf_bandwith
        # SVGD Los weight
        self.consistency_coef = cfg.train.consistency_coef
        self.kl_coef = cfg.train.kl_coef
        self.mean_sampling = cfg.train.mean_sampling

        self.q_target_max, self.q_target_min = cfg.train.q_target_max, cfg.train.q_target_min
        self.new_fraction = cfg.train.new_fraction
        self.q_norm = cfg.train.q_norm # "none", "q_grad", "svgd"; previous true - q_grad; false - none
    
        # repeat the target
        self.q_repeat = cfg.train.q_repeat
        
        # exploration related
        self.action_sample_noise = cfg.train.action_sample_noise
        self.sample_mode = cfg.train.sample_mode # "extrapolate"
        self.noise_mode = cfg.train.noise_mode  # "add"
        self.epsilon_ratio = cfg.train.epsilon_ratio  # 0.2
        self.min_extrp_step = cfg.train.min_extrp_step # 0
    
        self.q_batch_size = self.batch_size
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # configure Q norm regularization
        self.q_norm_max, self.q_norm_min, self.q_norm_reg = cfg.train.q_norm_max, cfg.train.q_norm_min, cfg.train.q_norm_reg

        # normalize the reward scale.
        # reward_scale < 0 activates EMARewardScaler; >= 0 uses a fixed multiplier.
        self.reward_scale = cfg.train.reward_scale
        if self.reward_scale < 0:
            ema_alpha = cfg.train.reward_scale_alpha # if hasattr(cfg.train, "reward_scale_alpha") else 0.05
            self.reward_scaler = EMARewardScaler(alpha=ema_alpha)
        else:
            self.reward_scaler = None
        
        # try loading model 
        self.load_model(os.path.join(cfg.logdir, "checkpoint"))

    def _ensure_finite(self, tensor, name):
        if torch.isfinite(tensor).all():
            return

        finite_vals = tensor[torch.isfinite(tensor)]
        if finite_vals.numel() == 0:
            raise RuntimeError(f"{name} contains no finite values.")

        raise RuntimeError(
            f"{name} contains non-finite values. "
            f"finite_min={finite_vals.min().item():.6f}, "
            f"finite_max={finite_vals.max().item():.6f}"
        )

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
    def update_flow_with_svgd(self, batch):
        """
        One policy update step.
        """
        state = batch["state"].to(self.device)
        batch_size = state.shape[0]
        ode_steps = self.model.ode_steps

        # ---------------------------------
        # 1. Compute SVGD direction
        # ---------------------------------
        state_sample = state.repeat_interleave(self.n_particles, dim=0)  # (M * P, C)
        # critic may receive time-augmented state; actor always uses raw state
        if "timestep" in batch:
            state_critic = torch.cat([state, batch["timestep"].to(self.device)], dim=-1)
            state_critic_sample = state_critic.repeat_interleave(self.n_particles, dim=0)
        else:
            state_critic_sample = state_sample

        _action_sample = self.model({"state": state_sample}, deterministic=False, forward_net=self.model.actor_ft)
        action_sample = _action_sample.trajectories
        action_sample_flat = action_sample.reshape(batch_size * self.n_particles, -1)

        # 1 is sufficient
        q_weighted = 1.0
        
        # ---------------------------------
        # 2. Flow matching update
        # ---------------------------------
        # compute the expected targets for point on trajectory
        B, _, _ = action_sample.shape
        ode_step = torch.randint(0, ode_steps, (B,), device=state.device)
        final_ind = (ode_step == ode_steps - 1).float() if self.svgd_type == "hybrid" else None
        t = ode_step.float() / ode_steps
        update_chain = _action_sample.chains[torch.arange(B, device=state.device), ode_step]  # (B, horizon, act)
        
        model_pred = self.model.actor_ft(update_chain, t, cond={"state": state_sample}) # (B * P, act), predict the velocity
        if self.model.prediction_type == "v_pred":
            exp_land_point = update_chain + (1 - t).view(B, 1, 1) * model_pred # compute the expected landing point   
        elif self.model.prediction_type == "x_pred":
            exp_land_point = model_pred
        
        # compute SVGD direction
        # state_sample_3d = state_sample.view(batch_size, self.n_particles, -1)
        state_critic_sample_3d = state_critic_sample.view(batch_size, self.n_particles, -1)
        action_sample_3d = action_sample.view(batch_size, self.n_particles, -1)
        exp_land_point_3d = exp_land_point.view(batch_size, self.n_particles, -1)

        if self.svgd_type in ["interp", "delta"]:
            phi, svgd_info = self.batch_svgd_direction(
                exp_land_point_3d,
                exp_land_point_3d,
                state_critic_sample_3d,
                self.q_net,
                final_ind = None)  # (M, P, C)
        else:
            phi, svgd_info = self.batch_svgd_direction(
                exp_land_point_3d,
                action_sample_3d,
                state_critic_sample_3d,
                self.q_net,
                final_ind = final_ind)  # (M, P, C)

        phi = phi.reshape(batch_size * self.n_particles, -1) # (M * P, C)

        # compute the teacher output on old samples
        old_t = torch.randint(0, self.model.ode_steps, (B,), device=state.device).float() / self.model.ode_steps # [0 - 0.5]
        old_t_view = old_t.view(B, 1, 1)
        old_x_t = (1 - old_t_view) * torch.randn_like(action_sample) + old_t_view * action_sample
        old_target = self.model.actor(old_x_t, old_t, cond={"state": state_sample})

        # update the model
        self.fm_optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            # the gradient is \nabla_x / ( 1 - t )
            updated_model_pred = self.model.actor_ft(update_chain, t, cond={"state": state_sample})
            dot_prod = torch.sum( updated_model_pred * phi.view_as(action_sample), dim = (-1, -2) )
            dot_prod = dot_prod / (ode_steps * (1 - t)) if self.model.prediction_type == "v_pred" else dot_prod / ode_steps # (B * P,)
            svgd_ft_loss = -1 * torch.mean(dot_prod) # torch.mean(weight * (updated_flow_prediction - updated_target) ** 2) 
            
            # consistency loss
            if self.model.prediction_type == "v_pred":
                exp_land_point_ = update_chain + (1 - t).view(B, 1, 1) * updated_model_pred # compute the expected landing point 
                consistency_loss = self.consistency_coef * torch.mean(  # weight the consistency by time, early step has small 
                    torch.sum((exp_land_point_ - action_sample) ** 2, dim = (-1, -2)) / ( (ode_steps * (1 - t) ) ** 2 ) 
                )
            elif self.model.prediction_type == "x_pred":
                exp_land_point_ = updated_model_pred
                consistency_loss = self.consistency_coef * torch.mean(
                    torch.sum((exp_land_point_ - action_sample) ** 2, dim = (-1, -2)) )
    
            # KL loss
            old_flow_prediction = self.model.actor_ft(old_x_t, old_t, cond={"state": state_sample})
            kl_loss = torch.mean( q_weighted * torch.sum((old_flow_prediction - old_target) ** 2, dim = (-1, -2)) ) * self.kl_coef # / self.n_particles
            
            ft_loss = svgd_ft_loss + kl_loss + consistency_loss
            ft_loss.backward()  

        self.fm_optimizer.step()
        self.fm_lr_scheduler.step()

        loss_info = {
            "fm_svgd_loss": svgd_ft_loss.item(),
            "kl_loss": kl_loss.item(),
            "consistency_loss": consistency_loss.item(),
        }
        loss_info.update(svgd_info)
        return loss_info
    
    def _compute_td_target(self, rewards, done, q_target, td_step, 
                            target_max = 1., target_min = 0.):
        """ Compute TD(n) targets 
        Args:
            rewards: (n_envs, time_step)
            done: (n_envs, time_step)
            q_target: (n_envs, time_step + 1)
            td_steps: int number of steps for TD(n)
        Return:
            td_q_targets: (n_envs, time_step)
        """
        n_envs, time_step = rewards.shape
        
        # Compute TD(n) targets
        td_q_targets = torch.zeros(n_envs, time_step, device=self.device)
        
        for t in range(time_step):
            # TD(n): look ahead up to td_step steps
            target_t_step = min(t + td_step, time_step)
            
            # Compute cumulative discounted rewards
            n_steps = target_t_step - t
            discount_factors = self.gamma ** torch.arange(n_steps, dtype=torch.float32, device=self.device)
            reward_slot = rewards[:, t:target_t_step]
            
            # Bootstrap with Q value at target step (with done mask)
            discounted_rewards = (reward_slot * discount_factors.unsqueeze(0)).sum(dim=-1)
            
            # TODO: The Done signal should change 
            bootstrap_value = (self.gamma ** n_steps) * q_target[:, target_t_step] * (1 - done[:, target_t_step - 1])
            
            # TD(n) target
            td_q_targets[:, t] = discounted_rewards + bootstrap_value
        
        return td_q_targets
    
    @torch.no_grad()
    def update_q_net_td(self, fresh_stat, td_steps:Union[int, List[int]] = 10, update_steps=10, num_sample_actions=4):
        """ Update the Q network using td(n) targets:
        Args:
            fresh_stat: dict of episode statistics
            td_steps: int or list of int, number of steps for TD(n)
            update_steps: number of gradient update steps
            num_sample_actions: number of sampled actions for target Q estimation
        """
        # compute the target
        states = torch.from_numpy(fresh_stat["states"]).to(self.device).float()
        actions = torch.from_numpy(fresh_stat["actions"]).to(self.device).float()
        n_envs, time_step, action_dim = fresh_stat["actions"].shape
        _, _, obs_dim = fresh_stat["states"].shape
        raw_rewards = torch.from_numpy(fresh_stat["rewards"]).to(self.device).float()
        done = torch.from_numpy(fresh_stat["done"]).to(self.device).float()

        if self.rollout_stage_cond:
            timesteps = torch.from_numpy(fresh_stat["timesteps"]).to(self.device).float()  # (n_envs, time_step)
            states_critic = torch.cat([states, timesteps.unsqueeze(-1)], dim=-1)
        else:
            states_critic = states

        # compute the policy actions (actor sees raw states)
        policy_actions_repeat = []
        for _ in range(num_sample_actions):
            policy_actions = self.model({"state": states.reshape(-1, obs_dim)}, deterministic=False, forward_net=self.model.actor_ft)
            policy_actions = policy_actions.trajectories[:, :self.act_steps, :].reshape(n_envs, time_step, -1)
            policy_actions_repeat.append(policy_actions)

        # apply reward scaling to raw rewards at training time
        if self.reward_scaler is not None:
            rewards = raw_rewards * self.reward_scaler.scale
        else:
            rewards = raw_rewards * self.reward_scale

        pbar = tqdm(range(update_steps), desc=f"Q TD update itr {self.itr}")
        for step in range(update_steps):
            # Get target Q values using the current Q network (no grad for target computation)
            q_target = torch.zeros_like(actions[..., 0])
            for policy_actions in policy_actions_repeat:
                _q_target, _ = self.q_net.compute_target(states_critic, policy_actions, reduction="min")
                _q_target = _q_target.reshape(n_envs, time_step)
                q_target += _q_target / num_sample_actions

            # clip the td targets to [0, 1] range
            q_target.clip_(max=self.q_target_max, min=self.q_target_min)

            # Compute TD(n) targets
            td_step = td_steps if isinstance(td_steps, int) else td_steps[step % len(td_steps)]
            td_q_targets = self._compute_td_target(rewards, done, q_target, td_step)
            self._ensure_finite(td_q_targets, "td_q_targets")

            # Update Q network
            self.q_optimizer.zero_grad()
            with torch.enable_grad():
                critic_loss, q_error, q_val = self.q_net.compute_loss(
                    states_critic[:, :-1, :], actions[:, :-1, :], td_q_targets, self.q_split_mode) # exclude terminal states
                self._ensure_finite(critic_loss.view(1), "critic_loss")
                critic_loss.backward()
            self.q_optimizer.step()

            pbar.set_postfix({"q_error": f"{q_error:.4f}", "q_value": f"{q_val:.4f}", "lr": f"{self.q_optimizer.param_groups[0]['lr']:.4f}"})
            pbar.update(1)
            if self.q_net.use_target: #  and step % 4 == 0:
                self.q_net.update_target_networks()
                    
        logging.info(f"Finished Q TD update with final q error: {q_error:.4f}, avg q value: {q_val:.4f}")        
        return q_error

    @torch.no_grad()
    def update_q_net_td_lambda(self, fresh_stat, td_lambda = 0.9, update_steps=10, num_sample_actions=4):
        # compute the target
        states = torch.from_numpy(fresh_stat["states"]).to(self.device).float()
        actions = torch.from_numpy(fresh_stat["actions"]).to(self.device).float()
        n_envs, time_step, action_dim = fresh_stat["actions"].shape
        _, _, obs_dim = fresh_stat["states"].shape
        raw_rewards = torch.from_numpy(fresh_stat["rewards"]).to(self.device).float()
        done = torch.from_numpy(fresh_stat["done"]).to(self.device).float()

        if self.rollout_stage_cond:
            timesteps = torch.from_numpy(fresh_stat["timesteps"]).to(self.device).float()  # (n_envs, time_step)
            states_critic = torch.cat([states, timesteps.unsqueeze(-1)], dim=-1)
        else:
            states_critic = states

        # compute the mask using raw rewards (q_target_max threshold is in raw reward space)
        mask = None

        # apply reward scaling at training time (buffer stores raw rewards)
        if self.reward_scaler is not None:
            rewards = raw_rewards * self.reward_scaler.scale
        else:
            rewards = raw_rewards * self.reward_scale

        # compute the policy actions (actor sees raw states)
        policy_actions_repeat = []
        for _ in range(num_sample_actions):
            policy_actions = self.model({"state": states.reshape(-1, obs_dim)}, deterministic=False, forward_net=self.model.actor_ft)
            policy_actions = policy_actions.trajectories[:, :self.act_steps, :].reshape(n_envs, time_step, -1)
            policy_actions_repeat.append(policy_actions)

        pbar = tqdm(range(update_steps), desc=f"Q TD update itr {self.itr}")
        for step in range(update_steps):
            self._check_cm()
            # Get target Q values using the current Q network (no grad for target computation)
            q_target = torch.zeros_like(actions[..., 0])
            for policy_actions in policy_actions_repeat:
                _q_target, _ = self.q_net.compute_target(states_critic, policy_actions, reduction="min")
                _q_target = _q_target.reshape(n_envs, time_step)
                q_target += _q_target / num_sample_actions

            # clip the td targets to [0, 1] range; (no reward scaler)
            q_target.clip_(max=self.q_target_max, min=self.q_target_min)

            # Compute TD(lambda) targets
            td_q_targets = torch.zeros(n_envs, time_step - 1, device=self.device)

            for t in reversed(range(time_step - 1)):
                # TD(lambda): recursive computation
                if t == time_step - 2:
                    td_q_targets[:, t] = rewards[:, t] + (1 - done[:, t]) * self.gamma * q_target[:, t + 1]
                else:
                    td_q_targets[:, t] = rewards[:, t] + (1 - done[:, t]) * self.gamma * \
                                        ( (1 - td_lambda) * q_target[:, t + 1] + td_lambda * td_q_targets[:, t + 1] )
            self._ensure_finite(td_q_targets, "td_lambda_q_targets")

            # Update Q network
            self.q_optimizer.zero_grad()
            with torch.enable_grad():
                critic_loss, q_error, q_val = self.q_net.compute_loss(
                    states_critic[:, :-1, :], actions[:, :-1, :], td_q_targets, self.q_split_mode, mask) # exclude terminal states
                self._ensure_finite(critic_loss.view(1), "critic_loss")
                critic_loss.backward()
            self.q_optimizer.step()

            pbar.set_postfix({"q_error": f"{q_error:.4f}", "q_value": f"{q_val:.4f}", "lambda": f"{td_lambda:.4f}", "lr": f"{self.q_optimizer.param_groups[0]['lr']:.4f}"})
            pbar.update(1)
            if self.q_net.use_target: # and step % 4 == 0:
                self.q_net.update_target_networks()
                    
        logging.info(f"Finished Q TD update with final q error: {q_error:.4f}, avg q value: {q_val:.4f}")
        return q_error

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0

        # reset env
        prev_obs_venv = self.reset_env_all()

        # trackers for episode returns and best step rewards within an episode, 
        # to be logged when an episode finishes 
        # (for furniture env, the episode return is already sparse reward, so no need to track best step reward)
        episode_returns = np.zeros(self.n_envs, dtype=np.float32)
        episode_max_step_rewards = np.full(self.n_envs, 0., dtype=np.float32)
        
        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)

            # Prepare per-env accumulators for episode summaries within this iteration
            completed_episode_returns = []
            completed_episode_best = []
            qvals = []
            
            fresh_stats = dict(
                states=np.empty((self.n_envs, self.n_steps + 1, self.cfg.obs_dim)), # terminal obs
                actions=np.empty((self.n_envs, self.n_steps + 1, self.cfg.action_dim * self.cfg.act_steps)),
                rewards=np.empty((self.n_envs, self.n_steps)),
                done=np.empty((self.n_envs, self.n_steps)),
                timesteps=np.zeros((self.n_envs, self.n_steps + 1)),
                # next_states=np.empty((self.n_envs, 0, self.cfg.obs_dim)),
                # next_actions=np.empty((self.n_envs, 0, self.cfg.action_dim * self.cfg.act_steps))
            )

            pbar = tqdm(range(self.n_steps), desc=f"Iteration {self.itr}")
            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                self._check_cm()

                # Select action
                with torch.no_grad():
                    cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device).reshape(-1, self.cfg.obs_dim)}
                    
                    # normal sampling
                    if eval_mode:
                        samples = self.model(cond=cond, deterministic=self.mean_sampling, forward_net=self.model.actor_ft)
                    else:
                        samples = self._train_sampling(cond)
                        
                    actions = samples.trajectories[:, :self.act_steps, :].reshape(self.n_envs, -1)
                    if self.rollout_stage_cond:
                        r_feat = torch.from_numpy(episode_returns).to(self.device).reshape(-1, 1).float() # (n_envs, 1)
                        state_for_q = torch.cat([cond["state"], r_feat], dim=-1)
                    else:
                        state_for_q = cond["state"]
                    cur_q_val, cur_q_std = self.q_net(state_for_q, actions, reduction="mean")
                    avg_q_val = cur_q_val.mean().item()
                    qvals.append(avg_q_val)
            
                    output_venv = (samples.trajectories.cpu().numpy())  # n_env x horizon x act dim
                    action_venv = output_venv[:, :self.act_steps, :]
                        
                # Apply multi-step action
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv

                # Accumulate rewards for current ongoing episodes and track best step reward
                episode_returns += reward_venv.astype(np.float32)
                episode_max_step_rewards = np.maximum(
                    episode_max_step_rewards, reward_venv.astype(np.float32)
                )
                
                # store the state info
                fresh_stats["states"][:, step, :] = prev_obs_venv["state"].reshape(self.n_envs, -1)
                fresh_stats["actions"][:, step, :] = action_venv.reshape(self.n_envs, -1)
                # TODO: Add normalization for the intrinsic reward from Q disagreement
                fresh_stats["rewards"][:, step] = reward_venv.reshape((self.n_envs, ))
                fresh_stats["done"][:, step] = done_venv.reshape((self.n_envs, ))

                # If an env finished an episode at this macro step, finalize and reset its accumulators
                for i in range(self.n_envs):
                    if done_venv[i]:
                        completed_episode_returns.append(float(episode_returns[i]))
                        if self.furniture_sparse_reward:
                            best_val = float(episode_returns[i])
                        else:
                            best_val = float(episode_max_step_rewards[i] / self.act_steps)
                        completed_episode_best.append(best_val)
                        episode_returns[i] = 0.0
                        episode_max_step_rewards[i] = -np.inf
                        
                # stage information for furniture
                fresh_stats["timesteps"][:, step + 1] = episode_returns # (step * self.act_steps) / self.max_episode_steps

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0
                pbar.set_postfix({
                    "mavg_return": f"{np.mean(episode_returns):.4f}"
                })
                pbar.update(1)
            pbar.close()
            
            # log the terminal 
            samples = self.model.train_sampling(
                cond={"state": torch.from_numpy(obs_venv["state"]).float().to(self.device)}, 
                deterministic=False,
                stop_idx=torch.randint(0, self.model.ode_steps, (self.n_envs,), device=self.device), 
                forward_net=self.model.actor_ft)
            actions = samples.trajectories[:, :self.act_steps, :].reshape(self.n_envs, -1).cpu().numpy()
            fresh_stats["states"][:, self.n_steps] = obs_venv["state"].reshape(self.n_envs, -1)
            fresh_stats["actions"][:, self.n_steps] = actions.reshape(self.n_envs, -1)
            # fresh_stats["timesteps"][:, self.n_steps] = episode_returns # (self.n_steps * self.act_steps) / self.max_episode_steps
            
            # if self.cfg.env.env_type == "furniture":
            #     # compute cumulative rewards for the episode, to be used for reward-to-go target in Q update
            #     fresh_stats["rewards"] = np.cumsum(fresh_stats["rewards"], axis=1) / self.n_steps

            # Summarize episode rewards for episodes that finished within this iteration
            if len(completed_episode_returns) > 0:
                episode_reward = np.array(completed_episode_returns, dtype=np.float32)
                episode_best_reward = np.array(completed_episode_best, dtype=np.float32)
                num_episode_finished = len(episode_reward)
                avg_episode_reward = float(np.mean(episode_reward))
                avg_best_reward = float(np.mean(episode_best_reward))
                success_rate = float(
                    np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
                )
            else:
                episode_reward = np.array([], dtype=np.float32)
                num_episode_finished = 0
                avg_episode_reward = 0.0
                avg_best_reward = 0.0
                success_rate = 0.0
                log.info("[WARNING] No episode completed within the iteration!")
            logging.info("avg #{:d} episode rewards: {:.4f} | success rate: {:.4f} |  avg_best_reward: {:.4f}"\
                            .format(num_episode_finished, avg_episode_reward, success_rate, avg_best_reward))

            # TODO Debug part; plot the target & pred curves for Q network
            # self._plot_q_target_pred(fresh_stats, [64, 32, 4, 2], 2, 16)

            # uncomment to visualize action tSNE
            # self.visualize_action(fresh_stats["states"][0])

            # update model
            total_q_loss, total_svgd_loss, total_kl_loss, total_consist_loss = [], [], [], []
            total_action_sample_dist, q_improves, grad_q_norm = [], [], []
            if not eval_mode:
                total_iterations = self.q_repeat
                total_trajs = self.n_envs # // len(self.q_net.q_nets)

                # Update EMA stats from raw rewards (do NOT scale here — raw rewards
                # are stored in the buffer; scaling is applied inside update_q_net_td*).
                if self.reward_scaler is not None:
                    self.reward_scaler.update(fresh_stats["rewards"])
                    logging.info(f"EMA reward scaler: ema={self.reward_scaler._ema:.4f}, scale={self.reward_scaler.scale:.4f}")
                
                # compute the Q value before 
                # q_val_bef = self._compute_q(fresh_stats)
                
                for i in range(total_iterations):
                    # combine half & half from fresh and old
                    combined_stats = {}
                    if len(self.old_stats["states"]) > 0:
                        # determine the number of fresh and old samples
                        ratio = max( 
                            len(fresh_stats["states"]) / (len(fresh_stats["states"]) + len(self.old_stats["states"])), self.new_fraction
                        )
                        # take the fresh
                        fresh_num = int(total_trajs * ratio)
                        fresh_idx = np.random.choice(
                            self.n_envs, size=fresh_num, replace=False
                        )
                        
                        old_nums = total_trajs - fresh_num
                        if self.weighted_sample: 
                            stages = np.sum(np.abs(self.old_stats["rewards"]), axis=-1).astype(np.int32) # (n_old_trajs, )
                            unique_stage = np.unique(stages)
                            
                            sample_num = [int(old_nums / len(unique_stage)) for _ in unique_stage] # more likely to sample from high reward trajs
                            sample_num[0] += old_nums - sum(sample_num) # adjust for rounding
                            old_idx = []
                            for stage_id, num in zip(unique_stage, sample_num):
                                stage_idx = np.where(stages == stage_id)[0]
                                if len(stage_idx) > 0:
                                    old_idx.append(
                                        np.random.choice(
                                            stage_idx, 
                                            size=num, 
                                            replace=True if len(stage_idx) < num else False
                                        )
                                    )
                            old_idx = np.concatenate(old_idx, axis=0)
                        else:
                            old_idx = np.random.choice(
                                self.old_stats["states"].shape[0], 
                                size=old_nums, 
                                replace=False
                            )
                        
                        # merge
                        for key in ["states", "actions", "rewards", "done", "timesteps"]:
                            combined_stats[key] = np.concatenate([fresh_stats[key][fresh_idx], self.old_stats[key][old_idx]], axis=0)
                    else:
                        fresh_idx = np.random.choice(
                            self.n_envs, size=total_trajs, replace=False
                        )
                        # select
                        for key in ["states", "actions", "rewards", "done", "timesteps"]:
                            combined_stats[key] = fresh_stats[key][fresh_idx]

                    # update Q network using TD(n) targets
                    cur_q_update_steps = self.q_update_steps // total_iterations
                    # TODO: Full td lambda range from 0 - 1.
                    td_lambda = self.td_lambda # if self.itr > self.n_critic_warmup_itr else 1.0 # 1 - 0
                    q_error = self.update_q_net_td_lambda(combined_stats, td_lambda=td_lambda, update_steps=cur_q_update_steps, num_sample_actions=self.q_bootstrap)
                    total_q_loss.append(q_error)
                    
                    # gradq_net_idx > 0; alternating between ensembles
                    # otherwise, -1 will default to average
                    self.gradq_net_idx = self.itr % len(self.q_net.q_nets) if self.gradq_net_idx >= 0 else -1 

                    # update flow matching policy with SVGD
                    if self.itr > self.n_critic_warmup_itr:
                        fm_update_steps = self.policy_update_steps // total_iterations
                        pbar = tqdm(range(fm_update_steps), desc=f"FM update itr {self.itr}")
                        for fm_iteration in range(fm_update_steps):
                            self._check_cm()
                            
                            start_idx = (fm_iteration) % self.n_particles
                            _states_np = combined_stats["states"][start_idx::self.n_particles, :-1, :]
                            batch_t = {
                                # filter out the last state and action for each traj
                                "state": torch.from_numpy(
                                    _states_np.reshape(-1, self.cfg.obs_dim)
                                ).float().to(self.device)
                            }
                            if self.rollout_stage_cond:
                                _ts_np = combined_stats["timesteps"][start_idx::self.n_particles, :-1]
                                batch_t["timestep"] = torch.from_numpy(
                                    _ts_np.reshape(-1, 1)
                                ).float().to(self.device)
                                
                            metrics = self.update_flow_with_svgd(batch_t)
                            total_svgd_loss.append(metrics["fm_svgd_loss"])
                            total_kl_loss.append(metrics["kl_loss"])
                            grad_q_norm.append(metrics["q_norm"])
                            total_consist_loss.append(metrics["consistency_loss"])
                            total_action_sample_dist.append(metrics["action_sample_distance"])
                            # logging.info(f"Flow matching SVGD loss: {metrics['fm_svgd_loss']:.4f}, svgd norm: {metrics['svgd_norm']:.4f}")
                            pbar.set_postfix({"svgd_loss": f"{metrics['fm_svgd_loss']:.4f}", "kl_loss": f"{metrics['kl_loss']:.4f}", "lr": f"{self.fm_optimizer.param_groups[0]['lr']:.4f}"})
                            pbar.update(1)
                        pbar.close()

                # q_val_aft = self._compute_q(fresh_stats)
                # q_improve = q_val_aft - q_val_bef
                # q_improves.append(q_improve)
            
                # merge for training buffer
                for key in ["states", "actions", "rewards", "done", "timesteps"]:
                    self.old_stats[key] = np.concatenate([self.old_stats[key], fresh_stats[key]], axis=0)
                    if len(self.old_stats[key]) > self.buffer_num_episode:
                        self.old_stats[key] = self.old_stats[key][-self.buffer_num_episode:]

                # copy the parameter from actor_ft to actor (in training mode)
                self.model.actor.load_state_dict(self.model.actor_ft.state_dict())
                self.model.actor.eval()
                for p in self.model.actor.parameters():
                    p.requires_grad_(False)
                    
            # Save model: numbered snapshot at save_model_freq, latest-only every other iteration
            if (self.itr % self.save_model_freq == 0 and self.itr > 0) or self.itr == self.n_train_itr - 1:
                self.save_model(save_numbered=True)

            # save the latest checkpoint
            self.save_model()
            torch.cuda.empty_cache()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            # if self.save_trajs:
            #     run_results[-1]["obs_full_trajs"] = obs_full_trajs
            #     run_results[-1]["obs_trajs"] = obs_trajs
            #     run_results[-1]["reward_trajs"] = reward_trajs
            
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    eval_stats = {
                            "eval/success rate": success_rate,
                            "eval/avg episode reward": avg_episode_reward,
                            "eval/avg best reward": avg_best_reward,
                            "eval/num episode": num_episode_finished,
                        }
                    if self.use_wandb:
                        wandb.log(
                            eval_stats,
                            step=self.itr,
                            commit=False,
                        )
                        
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    avg_q_loss_epoch = float(np.mean(total_q_loss)) if len(total_q_loss) > 0 else 0.0
                    avg_svgd_loss_epoch = float(np.mean(total_svgd_loss)) if len(total_svgd_loss) > 0 else 0.0
                    avg_q_improve = float(np.mean(q_improves)) if len(q_improves) > 0 else 0.0
                    kl_loss_epoch = float(np.mean(total_kl_loss)) if len(total_kl_loss) > 0 else 0.0
                    avg_grad_q_norm = float(np.mean(grad_q_norm)) if len(grad_q_norm) > 0 else 0.0
                    avg_consist_loss = float(np.mean(total_consist_loss)) if len(total_consist_loss) > 0 else 0.0
                    avg_action_sample_dist = float(np.mean(total_action_sample_dist)) if len(total_action_sample_dist) > 0 else 0.0
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | q loss {avg_q_loss_epoch:8.4f} |  svgd_loss {avg_svgd_loss_epoch:8.4f} | reward {avg_episode_reward:8.4f} | t:{time:8.4f}"
                    )
                    stats_dict = {
                        "total env step": cnt_train_step,
                        "train/avg episode reward": avg_episode_reward,
                        "train/num episode": num_episode_finished,
                        "train/success rate": success_rate,
                        "train/avg q loss": avg_q_loss_epoch,
                        "train/avg svgd loss": avg_svgd_loss_epoch,
                        "train/avg kl loss": kl_loss_epoch,
                        "train/q improvement": avg_q_improve,
                        "train/avg action sample dist": avg_action_sample_dist,
                        "train/avg q grad norm": avg_grad_q_norm,
                        "train/avg consistency loss": avg_consist_loss,
                        "train/svgd_temp": self.svgd_temp,
                        "Q network lr": self.q_optimizer.param_groups[0]["lr"],
                        "Flow matching policy lr": self.fm_optimizer.param_groups[0]["lr"],
                    }
                    if self.use_wandb:
                        wandb.log(stats_dict, step=self.itr, commit=True)
                    
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
    
    @torch.no_grad()    
    def _train_sampling(self, cond):
        """ Exploration Code for sampling actions during training """
        # normal sampling
        if self.sample_mode == "full":
            samples = self.model(cond=cond, deterministic=False, forward_net=self.model.actor_ft)
            gaussian_noise = self.action_sample_noise * torch.randn_like(samples.trajectories)  
            
            # add gaussian noise
            if self.noise_mode == 'add':
                samples.trajectories.add_(gaussian_noise)
            elif self.noise_mode == 'elbo':
                # extrapolate steps
                ode_step = torch.randint(self.min_extrp_step, self.model.ode_steps, (self.n_envs,), device=self.device)

                interp_point = samples.chains[:, self.min_extrp_step - 1, :, :]
                time_step = torch.empty(samples.chains.shape[0], device=samples.chains.device).fill_((self.min_extrp_step - 1) / self.model.ode_steps)
                pred = self.model.actor_ft(interp_point, time_step, cond=cond) # (B, A)
                prev_exp = interp_point + pred * (1 - time_step).view(-1, 1, 1) \
                        if self.model.prediction_type == 'v_pred' else pred # (B, A)
                
                samples_torch = torch.empty_like(samples.trajectories)

                for i in range(self.min_extrp_step, self.model.ode_steps):
                    interp_point = samples.chains[:, i, :, :]
                    time_step = torch.empty(samples.chains.shape[0], device=samples.chains.device).fill_(i / self.model.ode_steps)
                    pred = self.model.actor_ft(interp_point, time_step, cond=cond) # (B, A)
                    exp = interp_point + pred * (1 - time_step).view(-1, 1, 1) \
                            if self.model.prediction_type == 'v_pred' else pred # (B, A)
                    
                    distance = torch.norm(exp.view(self.n_envs, -1) - prev_exp.view(self.n_envs, -1), dim=-1) / 2 # compute the half-radius to previous action
                    cur_sample = exp + distance.view(-1, 1, 1) * torch.randn_like(exp)
                    samples_torch[ode_step == i] = cur_sample[ode_step == i] # copy only the sampled ones

                    # move to next
                    prev_exp = exp

                samples = Sample(samples_torch, None)
            elif self.noise_mode == 'min_extrp':
                end_point = samples.trajectories
                ode_step = torch.randint(self.min_extrp_step, self.model.ode_steps, (self.n_envs,), device=self.device)
                samples_torch = torch.empty_like(samples.trajectories)

                for i in range(self.min_extrp_step, self.model.ode_steps):
                    interp_point = samples.chains[:, i, :, :]
                    time_step = torch.empty(samples.chains.shape[0], device=samples.chains.device).fill_(i / self.model.ode_steps)
                    pred = self.model.actor_ft(interp_point, time_step, cond=cond) # (B, A)
                    exp = interp_point + pred * (1 - time_step).view(-1, 1, 1) \
                            if self.model.prediction_type == 'v_pred' else pred # (B, A)
                    
                    distance = torch.norm(end_point.view(self.n_envs, -1) - exp.view(self.n_envs, -1), dim=-1) # compute the half-radius to previous action
                    cur_sample = end_point + distance.view(-1, 1, 1) * torch.randn_like(exp)
                    samples_torch[ode_step == i] = cur_sample[ode_step == i] # copy only the sampled ones

                samples = Sample(samples_torch, None)
                
            elif self.noise_mode == "qsample" and self.itr > self.n_critic_warmup_itr:
                # do gradient ascent on the final samples
                # add small noise, similar to the Langevin MCMC.
                action = samples.trajectories  # (B, act_steps, action_dim)
                state = cond["state"]  # (B, obs_dim)

                # reshape to (B, 1, total_A) for compute_gradq
                B = action.shape[0]
                action_flat = action.reshape(B, 1, -1)  # (B, 1, act_steps*action_dim)
                state_rep = state.unsqueeze(1)  # (B, 1, obs_dim)

                # 4-step Langevin update: x = x + step_size * grad_Q(x) + noise_scale * eps
                step_size = self.action_sample_noise
                for _ in range(4):
                    grad_q = self.q_net.compute_gradq(state_rep, action_flat, self.gradq_net_idx)  # (B, 1, total_A)
                    noise = torch.randn_like(action_flat)
                    action_flat = action_flat + step_size * grad_q + step_size * noise / 2

                samples = Sample(action_flat.reshape_as(action), None)
                       
        # sample & select using the Q network
        elif self.sample_mode == "mppi":
            # extrapolate sampling
            # ode_step = torch.randint(self.min_extrp_step, self.model.ode_steps, (states.shape[0],), device=self.device)
            samples = self.model(cond=cond, deterministic=False, forward_net=self.model.actor_ft)
            
            # take number of action particles
            total_action_dim = self.cfg.action_dim * self.cfg.act_steps
            states = cond["state"].repeat_interleave(total_action_dim, dim=0)
            
            # add noise to sampled actions
            predicted_actions = samples.trajectories.reshape(self.n_envs, -1) # (B, C)
            predicted_actions = predicted_actions.repeat_interleave(total_action_dim, dim=0)
            gaussian_noise = self.action_sample_noise * torch.randn_like(predicted_actions)
            predicted_actions.add_(gaussian_noise)
            
            sample_nums = 10
            # update mean and covariance            
            candidate_actions = predicted_actions.reshape(-1, total_action_dim, total_action_dim).clone()
            states = states.reshape(-1, total_action_dim, self.cfg.obs_dim)
            for i in range(sample_nums):
                mean, L = self.q_net.compute_mean_var(states, candidate_actions, self.svgd_temp)
        
                noise = torch.randn_like(candidate_actions)
                candidate_actions = mean[:, None, :] + torch.einsum('ikc,ijc->ijk', L, noise)
                candidate_actions = candidate_actions.contiguous()
            
            # just randomly pick the first one
            candidate_actions = candidate_actions[:, 0, :].reshape(self.n_envs, self.cfg.act_steps, self.cfg.action_dim)
            samples = Sample(candidate_actions, None)

        elif self.sample_mode == "extrapolate":
            # extrapolate sampling
            ode_step = torch.randint(self.min_extrp_step, self.model.ode_steps, (self.n_envs,), device=self.device)
            samples = self.model.train_sampling(
                cond=cond, stop_idx=ode_step, 
                deterministic=False, random_noise=self.action_sample_noise, forward_net=self.model.actor_ft)
        else:
            raise NotImplementedError(f"{self.sample_mode} not implemented")
            
        # gaussian noisy exploration
        return samples
            
    @torch.no_grad()   
    def _plot_q_target_pred(self, stats, td_steps = [32], num_sample_actions = 2, num_samples = 8):
        """ Debug function to plot the Q target vs Q pred """
        rewards =  torch.from_numpy(stats["rewards"]).to(self.device).float()
        success = (rewards.sum(-1) >= 1)
        success_id = torch.nonzero(success).view(-1)
        fail_id = torch.nonzero(~success).view(-1)
    
        # compute the policy actions
        states = torch.from_numpy(stats["states"]).to(self.device).float()
        actions = torch.from_numpy(stats["actions"]).to(self.device).float()
        dones = torch.from_numpy(stats["done"]).to(self.device).float()
        n_envs, time_step, obs_dim = states.shape
        policy_actions_repeat = []
        for _ in range(num_sample_actions):
            policy_actions = self.model({"state": states.reshape(-1, obs_dim)}, deterministic=False, forward_net=self.model.actor_ft)
            policy_actions = policy_actions.trajectories[:, :self.act_steps, :].reshape(n_envs, time_step, -1)
            policy_actions_repeat.append(policy_actions)
        
        # Get target Q values using the current Q network (no grad for target computation)
        q_targets = torch.zeros_like(states[..., 0])
        for policy_actions in policy_actions_repeat:
            _q_target, _ = self.q_net.compute_target(states, policy_actions, reduction="min")
            # _q_target = q_target_pred[0] if self.q_net.double_q else q_target_pred
            _q_target = _q_target.reshape(n_envs, time_step)
            q_targets += _q_target / num_sample_actions
        
        # iterate over td steps
        for td_step in td_steps:
            td_q_target = self._compute_td_target(rewards, dones, q_targets, td_step=td_step)
            q_pred, _ = self.q_net(states, actions, reduction="mean")
            
            n_col = 4
            n_row = math.ceil(num_samples / n_col)
            fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_row, 4 * n_col))
            for i in range(num_samples):
                # fix error when success num < 8
                index_ids = success_id if (i < num_samples // 2  and i < len(success_id)) else fail_id
                # find the axes
                ax = axes[i // 4, i % 4]
                ax.plot(td_q_target[index_ids[i]].detach().cpu().numpy(), label="TD Target", color='blue')
                ax.plot(q_pred[index_ids[i]].detach().cpu().numpy(), label="Q Pred", color='orange')
                ax.set_title(f"Sample {index_ids[i]}")
                ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(self.checkpoint_dir, "../", f"itr-{self.itr}_td{td_step}_pred.png"))

            # close the figure
            plt.close(fig)

    def visualize_action(self, rollout_states, num_actions=512, save_dir=None, tag=""):
        """
        For each state in a rollout, sample actions, compute Q values and Q gradients,
        then project everything onto a t-SNE map with gradient arrows and Q-value coloring.

        Args:
            rollout_states: np.ndarray or torch.Tensor of shape (L, S)
            num_actions:    number of action samples per state (default 512)
            save_dir:       directory to save PNG files (defaults to <checkpoint_dir>/../action_tsne)
            tag:            optional string appended to each filename
        """
        from sklearn.manifold import TSNE

        if save_dir is None:
            save_dir = os.path.join(self.checkpoint_dir, "..", "action_tsne")
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(rollout_states, np.ndarray):
            rollout_states = torch.from_numpy(rollout_states).float().to(self.device)
        else:
            rollout_states = rollout_states.float().to(self.device)

        L, obs_dim = rollout_states.shape
        total_action_dim = self.act_steps * self.cfg.action_dim

        for step_idx in range(L):
            state = rollout_states[step_idx]  # (S,)
            states_rep = state.unsqueeze(0).expand(num_actions, -1).contiguous()  # (num_actions, S)

            # --- sample actions from the policy ---
            with torch.no_grad():
                cond = {"state": states_rep}
                samples = self.model(cond, deterministic=False, forward_net=self.model.actor_ft)
                actions = samples.trajectories[:, :self.act_steps, :].reshape(num_actions, total_action_dim)

                # Average pairwise distance between action samples
                avg_pairwise_dist = torch.cdist(actions.unsqueeze(0), actions.unsqueeze(0)).mean().item()

                # --- Q values ---
                q_vals, _ = self.q_net(states_rep, actions, reduction="mean")  # (num_actions,)

            q_vals_np = q_vals.cpu().numpy()
            actions_np = actions.cpu().numpy()

            # --- Q gradients (compute_gradq uses torch.enable_grad internally) ---
            grad_q = self.q_net.compute_gradq(states_rep, actions)  # (num_actions, A)
            grad_q_np = grad_q.cpu().numpy()

            # Normalize gradient directions for probe-point construction
            grad_norm = np.linalg.norm(grad_q_np, axis=-1, keepdims=True) + 1e-8
            grad_unit = grad_q_np / grad_norm

            # Probe points displaced by a fraction of the action-space spread
            epsilon = 0.05 * (actions_np.max() - actions_np.min() + 1e-8)
            probe_actions_np = actions_np + epsilon * grad_unit

            # --- joint t-SNE on originals + probes ---
            combined = np.concatenate([actions_np, probe_actions_np], axis=0)  # (2*N, A)
            perplexity = min(30, num_actions // 5)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
            embedded = tsne.fit_transform(combined)  # (2*N, 2)

            emb_orig  = embedded[:num_actions]   # (N, 2)
            emb_probe = embedded[num_actions:]   # (N, 2)

            # Arrow directions in t-SNE space, rescaled to a fixed visual length
            arrows = emb_probe - emb_orig        # (N, 2)
            arrow_norm = np.linalg.norm(arrows, axis=-1, keepdims=True) + 1e-8
            spread = np.std(emb_orig) + 1e-8
            arrow_scale = 0.08 * spread          # fixed visual arrow length
            arrows_vis = (arrows / arrow_norm) * arrow_scale

            # --- plot ---
            fig, ax = plt.subplots(figsize=(16, 14))

            sc = ax.scatter(
                emb_orig[:, 0], emb_orig[:, 1],
                c=q_vals_np, cmap="viridis", s=20, alpha=0.85, zorder=2,
            )
            plt.colorbar(sc, ax=ax, label="Q value")

            ax.quiver(
                emb_orig[:, 0], emb_orig[:, 1],
                arrows_vis[:, 0], arrows_vis[:, 1],
                q_vals_np, cmap="viridis",
                angles="xy", scale_units="xy", scale=1,
                alpha=0.6, width=0.003, zorder=3,
            )

            ax.set_title(
                f"Action t-SNE — step {step_idx:04d} | "
                f"Q [{q_vals_np.min():.3f}, {q_vals_np.max():.3f}] | "
                f"avg pairwise dist {avg_pairwise_dist:.4f}"
            )
            ax.set_xlabel("t-SNE dim 1")
            ax.set_ylabel("t-SNE dim 2")
            ax.set_aspect("equal")

            suffix = f"_{tag}" if tag else ""
            fname = os.path.join(save_dir, f"step_{step_idx:04d}{suffix}.png")
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved action t-SNE plot: {fname}")

    def save_model(self, save_numbered=False):
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "q_net": self.q_net.state_dict(),
            "q_optimizer_state": self.q_optimizer.state_dict(),
            "fm_optimizer_state": self.fm_optimizer.state_dict(),
            "fm_scheduler": self.fm_lr_scheduler.state_dict(),
            "reward_scaler": self.reward_scaler.state_dict() if self.reward_scaler is not None else None,
        }  # right now `model` includes weights for `network`, `actor`, `actor_ft`.

        # additionally save a numbered snapshot for archival
        if save_numbered:
            savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, savepath)
            log.info(f"Saved model to {savepath}")
            np.save(os.path.join(self.checkpoint_dir, f"buffer_{self.itr}.npy"), self.old_stats)
        else:
            # always overwrite the rolling "latest" files for crash recovery
            for fname in ("state_latest.pt", "buffer_latest.npy"):
                fpath = os.path.join(self.checkpoint_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
            
            torch.save(data, os.path.join(self.checkpoint_dir, "state_latest.pt"))
            np.save(os.path.join(self.checkpoint_dir, "buffer_latest.npy"), self.old_stats)
            log.info(f"Saved latest checkpoint at iteration {self.itr}")
            
    def load_model(self, load_path):
        # Prefer the rolling "latest" checkpoint saved every iteration
        latest_model_path = os.path.join(load_path, "state_latest.pt")
        latest_buffer_path = os.path.join(load_path, "buffer_latest.npy")
        if not os.path.exists(latest_model_path):
            # Fall back to numbered checkpoints
            pt_files = glob.glob(os.path.join(load_path, "state_*.pt"))
            epochs = []
            for file in pt_files:
                basename = os.path.basename(file)
                epoch_str = basename.split('_')[1].split('.')[0]
                try:
                    epochs.append(int(epoch_str))
                except ValueError:
                    pass
            if len(epochs) == 0:
                log.warning(f"No model files found in {load_path}. Starting from scratch.")
                return
            latest_epoch = max(epochs)
            latest_model_path = os.path.join(load_path, f"state_{latest_epoch}.pt")
            latest_buffer_path = os.path.join(load_path, f"buffer_{latest_epoch}.npy")

        checkpoint = torch.load(latest_model_path, map_location=self.device)

        # load the model state
        self.model.load_state_dict(checkpoint["model"])
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state"])
        self.fm_optimizer.load_state_dict(checkpoint["fm_optimizer_state"])
        self.fm_lr_scheduler.load_state_dict(checkpoint["fm_scheduler"])
        if self.reward_scaler is not None and checkpoint.get("reward_scaler") is not None:
            self.reward_scaler.load_state_dict(checkpoint["reward_scaler"])
        self.itr = checkpoint["itr"] + 1 # start from the next iteration
        log.info(f"Loaded model from {latest_model_path} at iteration {self.itr}")

        # for q_net in self.q_net.q_nets:
        #     self.q_net._init_module(q_net)

        # load the corresponding buffer
        if os.path.exists(latest_buffer_path):
            self.old_stats = np.load(latest_buffer_path, allow_pickle=True).item()
            # backward compat: older buffers may lack timesteps
            if "timesteps" not in self.old_stats:
                n_buf = len(self.old_stats["states"])
                self.old_stats["timesteps"] = np.zeros((n_buf, self.n_steps + 1))
            log.info(f"Loaded replay buffer from {latest_buffer_path} with {len(self.old_stats['states'])} transitions.")
        else:
            log.warning(f"No replay buffer file found at {latest_buffer_path}. Starting with empty buffer.")
        
