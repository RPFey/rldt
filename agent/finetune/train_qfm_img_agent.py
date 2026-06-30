### Flow Matching with Q Learning related algorithm
# QAM - arxiv Link:
# SDAC - arxiv link: https://arxiv.org/pdf/2502.00361
# FQL - arxiv link:  https://arxiv.org/abs/2502.02538

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Union, List
from torch.distributions import Normal
from agent.finetune.train_agent import TrainAgent
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

class QImgAgent(TrainAgent):
    """ Flow matching with Q learning for image-based control (base class).
        SVGD | QAM | FQL | SDAC — implement _update_actor to change algorithm"""
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
        
        # data augmentation for image observation; Apply only to critic
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)
        
        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        if hasattr(self.model.critic, "Q2"):
            use_target = True
            del self.model.critic.Q2  # only use single Q for FM-SVGD
        else:
            use_target = False
        self.q_net:QImgEnsemble = QImgEnsemble(self.model.critic, self.obs_dims, cfg.train.ensemble_q, use_target, cfg.train.q_init_std, cfg.train.q_target_mom)

        # Initialize critic backbone from pretrained actor backbone
        actor_backbone_state = self.model.actor_ft.backbone.state_dict()
        self.q_net.feat_encoder.backbone.load_state_dict(actor_backbone_state)
        if self.q_net.use_target:
            self.q_net.target_encoder.backbone.load_state_dict(actor_backbone_state)
            
        self.q_optimizer = torch.optim.AdamW(
            [
                {"params": self.q_net.q_nets.parameters(), "lr": cfg.train.q_update_lr},
                {"params": self.q_net.feat_encoder.parameters(), "lr": cfg.train.q_backbone_lr},
            ],
            weight_decay=cfg.train.critic_weight_decay
        )
        
        self.gamma = cfg.train.gamma
        self.q_update_steps = cfg.train.q_update_steps
        self.weighted_sample = cfg.train.weighted_sample
        self.policy_update_steps = cfg.train.policy_update_steps
        self.gradq_net_idx = cfg.train.gradq_net_idx
        self.q_bootstrap = cfg.train.q_bootstrap
        self.td_lambda = cfg.train.td_lambda
        self.q_mask = cfg.train.q_mask
        self.mean_sampling = cfg.train.mean_sampling
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        self.q_target_max, self.q_target_min = cfg.train.q_target_max, cfg.train.q_target_min
        self.new_fraction = cfg.train.new_fraction
        self.actor_ema_mom = cfg.train.actor_ema_mom
    
        # repeat the target
        self.q_repeat = cfg.train.q_repeat
        
        # exploration related
        self.action_sample_noise = cfg.train.action_sample_noise
        self.sample_mode = cfg.train.sample_mode # "extrapolate"
        self.noise_mode = cfg.train.noise_mode  # "add"
        self.epsilon_ratio = cfg.train.epsilon_ratio  # 0.2
        self.min_extrp_step = cfg.train.min_extrp_step # 0
        
        self.mini_batch_split = getattr(cfg.train, "mini_batch_split", 1)
        self.episode_factor = self.n_steps * self.act_steps // self.cfg.env.max_episode_steps 
        self.num_traj_per_chunk = 4 * self.episode_factor
        self.num_trajs = cfg.train.num_trajs * self.episode_factor
        self.buffer_num_episode = cfg.train.buffer_num_episode * self.n_envs * self.episode_factor # // 2 # half for success and half for failure
        
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
        
    def _update_actor(self, step_idx, batch_t, loss_scale=1.0):
        raise NotImplementedError("Actor update is handled separately in the main training loop for QFM-SVGD.")

    @torch.no_grad()
    def update_q_net_td_lambda(self, fresh_stat, td_lambda=0.9, update_steps=10, num_sample_actions=4):
        # unpack obs dict from buffer
        obs_tensors = {
            k: torch.from_numpy(fresh_stat["states"][k]).to(self.device).float()
            for k in self.obs_dims
        }  # each: (n_envs, T, *obs_shape)
        actions = torch.from_numpy(fresh_stat["actions"]).to(self.device).float()
        n_envs, time_step, _ = fresh_stat["actions"].shape
        raw_rewards = torch.from_numpy(fresh_stat["rewards"]).to(self.device).float()
        done = torch.from_numpy(fresh_stat["done"]).to(self.device).float()

        if self.augment:
            s, t, c, h, w = obs_tensors["rgb"].shape
            rgb = einops.rearrange(
                obs_tensors["rgb"],
                "s t c h w -> (s t) c h w",
            )
            rgb = self.aug(rgb)
            obs_tensors["rgb"] = einops.rearrange(
                rgb,
                "(s t) c h w -> s t c h w",
                s=s, t=t
            )
        
        # compute the mask using raw rewards (q_target_max threshold is in raw reward space)
        if self.q_mask:
            episode_cum_rewards = torch.zeros_like(raw_rewards)
            episode_cum_rewards[:, 0] = raw_rewards[:, 0]
            for t in range(1, time_step - 1):
                episode_cum_rewards[:, t] = (1 - done[:, t-1]) * episode_cum_rewards[:, t-1] + raw_rewards[:, t]
            complete_mask = (episode_cum_rewards >= self.best_reward_threshold_for_success) & (raw_rewards == 0)
            mask = ~complete_mask
            done = ((done > 0.) | (episode_cum_rewards >= self.best_reward_threshold_for_success)).float()
        else:
            mask = None

        # apply reward scaling at training time (buffer stores raw rewards)
        if self.reward_scaler is not None:
            rewards = raw_rewards * self.reward_scaler.scale
        else:
            rewards = raw_rewards * self.reward_scale

        # compute the policy actions
        policy_actions_repeat = []
        for _ in range(num_sample_actions):
            policy_actions, _ = self.model.cache_sampling(_flat_cond(obs_tensors, self.obs_dims), deterministic=False, forward_net=self.model.actor_ft)
            policy_actions = policy_actions.trajectories[:, :self.act_steps, :].reshape(n_envs, time_step, -1)
            policy_actions_repeat.append(policy_actions)

        pbar = tqdm(range(update_steps), desc=f"Q TD update itr {self.itr}")
        for step in range(update_steps):
            self._check_cm()
            
            # Get target Q values using the current Q network (no grad for target computation)
            q_target = torch.zeros_like(actions[..., 0])
            for policy_actions in policy_actions_repeat:
                _q_target, _ = self.q_net.compute_target(_flat_cond(obs_tensors, self.obs_dims), policy_actions, reduction="min")
                _q_target = _q_target.reshape(n_envs, time_step)
                q_target += _q_target / num_sample_actions

            # clip the td targets to [0, 1] range
            q_target.clip_(max=self.q_target_max, min=self.q_target_min)

            # Compute TD(lambda) targets
            td_q_targets = torch.zeros(n_envs, time_step - 1, device=self.device)
            for t in reversed(range(time_step - 1)):
                if t == time_step - 2:
                    td_q_targets[:, t] = rewards[:, t] + (1 - done[:, t]) * self.gamma * q_target[:, t + 1]
                else:
                    td_q_targets[:, t] = rewards[:, t] + (1 - done[:, t]) * self.gamma * (
                        (1 - td_lambda) * q_target[:, t + 1] + td_lambda * td_q_targets[:, t + 1]
                    )
            self._ensure_finite(td_q_targets, "td_lambda_q_targets")

            # Update Q network (exclude terminal states)
            obs_no_terminal = {k: obs_tensors[k][:, :-1] for k in self.obs_dims}
            actions_no_terminal = actions[:, :-1, :]
            B = actions_no_terminal.shape[0]
            chunk = B // self.mini_batch_split
            self.q_optimizer.zero_grad()
            q_error_accum, q_val_accum = 0.0, 0.0
            with torch.enable_grad():
                for i in range(self.mini_batch_split):
                    sl = slice(i * chunk, (i + 1) * chunk if i < self.mini_batch_split - 1 else B)
                    obs_chunk = {k: obs_no_terminal[k][sl] for k in self.obs_dims}
                    critic_loss, q_error, q_val = self.q_net.compute_loss(
                        obs_chunk, actions_no_terminal[sl], td_q_targets[sl],
                        mask[sl] if mask is not None else None)
                    self._ensure_finite(critic_loss.view(1), "critic_loss")
                    (critic_loss / self.mini_batch_split).backward()
                    q_error_accum += q_error / self.mini_batch_split
                    q_val_accum += q_val / self.mini_batch_split
            q_error, q_val = q_error_accum, q_val_accum
            self.q_optimizer.step()

            pbar.set_postfix({"q_error": f"{q_error:.4f}", "q_value": f"{q_val:.4f}", "lambda": f"{td_lambda:.4f}", "lr": f"{self.q_optimizer.param_groups[0]['lr']:.4f}"})
            pbar.update(1)
            if self.q_net.use_target:
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
        
        prefetcher = HDF5Fetcher(self.checkpoint_dir, max_buffer_num=self.buffer_num_episode)
        prefetcher.prefetch(
            self.num_trajs - int(self.num_trajs * self.new_fraction)
        )  # start loading before loop

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
            
            episode_length = self.cfg.env.max_episode_steps  // self.act_steps
            fresh_stats = dict(
                states={
                    k: np.empty((0, episode_length + 1, *shape)) for k, shape in self.obs_dims.items()
                },
                actions=np.empty((0, episode_length + 1, self.cfg.action_dim * self.cfg.act_steps)),
                rewards=np.empty((0, episode_length)),
                done=np.empty((0, episode_length)),
            )

            timer(reset=True)
            pbar = tqdm(range(self.n_steps), desc=f"Iteration {self.itr}")
            for _ in range(self.episode_factor):
                
                fresh_stats_ = dict(
                    states={
                        k: np.empty((self.n_envs, episode_length + 1, *shape)) for k, shape in self.obs_dims.items()
                    },
                    actions=np.empty((self.n_envs, episode_length + 1, self.cfg.action_dim * self.cfg.act_steps)),
                    rewards=np.empty((self.n_envs, episode_length)),
                    done=np.empty((self.n_envs, episode_length)),
                ) # for a complete episode

                # Collect a set of trajectories from env
                for step in range(episode_length):
                    self._check_cm()

                    # Select action
                    with torch.no_grad():
                        
                        # store in (C, H, W)
                        if 'rgb' in prev_obs_venv and prev_obs_venv['rgb'].shape[-1] == 3:
                            dims = list(range(prev_obs_venv['rgb'].ndim))
                            dims[-1] = dims[-2]; dims[-2] = dims[-3]; dims[-3] = prev_obs_venv['rgb'].ndim - 1
                            prev_obs_venv['rgb'] = np.transpose(prev_obs_venv['rgb'], dims)

                        cond = {
                            k: torch.from_numpy(prev_obs_venv[k]).float().to(self.device)
                            for k in self.obs_dims
                        }
                        
                        # normal sampling
                        if eval_mode:
                            samples, _ = self.model.cache_sampling(cond=cond, deterministic=self.mean_sampling, forward_net=self.model.actor_ft)
                        else:
                            samples = self._train_sampling(cond)
                            
                        actions = samples.trajectories[:, :self.act_steps, :].reshape(self.n_envs, -1)
                        cur_q_val, cur_q_std = self.q_net(cond, actions, reduction="mean")
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
                    
                    # store the obs info (state + rgb)
                    for k in self.obs_dims:
                        fresh_stats_["states"][k][:, step] = prev_obs_venv[k].reshape(self.n_envs, *self.obs_dims[k])
                    fresh_stats_["actions"][:, step, :] = action_venv.reshape(self.n_envs, -1)
                    # TODO: Add normalization for the intrinsic reward from Q disagreement
                    fresh_stats_["rewards"][:, step] = reward_venv.reshape((self.n_envs, )) 
                    fresh_stats_["done"][:, step] = done_venv.reshape((self.n_envs, ))

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

                    # update for next step
                    prev_obs_venv = obs_venv

                    # count steps --- not acounting for done within action chunk
                    cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0
                    pbar.set_postfix({
                        "mavg_return": f"{np.mean(episode_returns):.4f}"
                    })
                    pbar.update(1)
                
                # log the terminal obs and a sampled action
                terminal_cond = {
                    k: torch.from_numpy(obs_venv[k]).float().to(self.device)
                    for k in self.obs_dims
                }
                samples = self.model.train_sampling(
                    cond=terminal_cond,
                    deterministic=False,
                    stop_idx=torch.randint(0, self.model.ode_steps, (self.n_envs,), device=self.device),
                    forward_net=self.model.actor_ft)
                actions = samples.trajectories[:, :self.act_steps, :].reshape(self.n_envs, -1).cpu().numpy()
                for k in self.obs_dims:
                    fresh_stats_["states"][k][:, episode_length] = obs_venv[k].reshape(self.n_envs, *self.obs_dims[k])
                fresh_stats_["actions"][:, episode_length] = actions.reshape(self.n_envs, -1)

                # Post-processing - get rewards for change [Robomimic]
                rewards_diff = np.zeros_like(fresh_stats_["rewards"])
                rewards_diff[:, 0] = fresh_stats_["rewards"][:, 0]
                rewards_diff[:, 1:] = fresh_stats_["rewards"][:, 1:] - (1 - fresh_stats_["done"][:, :-1]) * fresh_stats_["rewards"][:, :-1] 
                rewards_diff /= max(1, fresh_stats_["rewards"].max()) # (1 - complete)
                fresh_stats_["rewards"] = rewards_diff
                
                # Concatenate with the fresh stats from previous episodes within this iteration
                fresh_stats["rewards"] = np.concatenate([fresh_stats["rewards"], fresh_stats_["rewards"]], axis=0)
                fresh_stats["done"] = np.concatenate([fresh_stats["done"], fresh_stats_["done"]], axis=0)
                fresh_stats["actions"] = np.concatenate([fresh_stats["actions"], fresh_stats_["actions"]], axis=0)
                for k in self.obs_dims:
                    fresh_stats["states"][k] = np.concatenate([fresh_stats["states"][k], fresh_stats_["states"][k]], axis=0)
                
            rollout_end = timer() # time for one iteration rollout
            pbar.close()
            
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

            # update model
            total_q_loss, total_actor_loss, total_kl_loss, total_consist_loss = [], [], [], []
            total_action_sample_dist, grad_q_norm = [], []
            if not eval_mode:
                total_iterations = self.q_repeat
                num_fresh_stats = fresh_stats["rewards"].shape[0]
                total_trajs = self.num_trajs # self.n_envs # // len(self.q_net.q_nets)

                # Update EMA stats from raw rewards (do NOT scale here — raw rewards
                # are stored in the buffer; scaling is applied inside update_q_net_td*).
                if self.reward_scaler is not None:
                    self.reward_scaler.update(fresh_stats["rewards"])
                    logging.info(f"EMA reward scaler: ema={self.reward_scaler._ema:.4f}, scale={self.reward_scaler.scale:.4f}")
                
                # compute the Q value before 
                # q_val_bef = self._compute_q(fresh_stats)

                for i in range(total_iterations):
                    # retrieve all chunk files and combine with fresh stats; sample a subset of trajectories if needed
                    start_io = time_m.time()
                    old_stats = prefetcher.get()
                    prefetcher.prefetch(total_trajs - int(total_trajs * self.new_fraction))
                    n_old = len(old_stats["actions"])
                    log.info(f"File I/O {n_old} Trajs wait time {time_m.time() - start_io}s")

                    # combine half & half from fresh and old
                    combined_stats = {}
                    if n_old > 0:
                        # determine the number of fresh and old samples
                        fresh_num = int(total_trajs * self.new_fraction)
                        fresh_idx = np.random.choice(
                            num_fresh_stats, size=fresh_num, replace=False
                        )
                        
                        # merge: handle states dict separately
                        combined_stats["states"] = {
                            k: np.concatenate([fresh_stats["states"][k][fresh_idx], old_stats["states"][k]], axis=0)
                            for k in self.obs_dims
                        }
                        for key in ["actions", "rewards", "done"]:
                            combined_stats[key] = np.concatenate([fresh_stats[key][fresh_idx], old_stats[key]], axis=0)
                    else:
                        fresh_idx = np.random.choice(num_fresh_stats, size=total_trajs, replace=False)
                        combined_stats["states"] = {
                            k: fresh_stats["states"][k][fresh_idx] for k in self.obs_dims
                        }
                        for key in ["actions", "rewards", "done"]:
                            combined_stats[key] = fresh_stats[key][fresh_idx]

                    # update Q network using TD(n) targets
                    cur_q_update_steps = self.q_update_steps // total_iterations
                    q_error = self.update_q_net_td_lambda(combined_stats, td_lambda=self.td_lambda, update_steps=cur_q_update_steps, num_sample_actions=self.q_bootstrap)
                    total_q_loss.append(q_error)
                    
                    # gradq_net_idx > 0; alternating between ensembles
                    # otherwise, -1 will default to average
                    self.gradq_net_idx = self.itr % len(self.q_net.q_nets) if self.gradq_net_idx >= 0 else -1 

                    # update flow matching policy
                    if self.itr > self.n_critic_warmup_itr:
                        fm_batch_size = combined_stats["actions"].shape[0] // 8
                        fm_update_steps = self.policy_update_steps // total_iterations
                        pbar = tqdm(range(fm_update_steps), desc=f"FM update itr {self.itr}")
                        for fm_iteration in range(fm_update_steps):
                            self._check_cm()
                            
                            # filter out the last timestep (terminal) for each traj
                            # Sample index
                            batch_idx = np.random.choice(
                                combined_stats["actions"].shape[0], size=fm_batch_size, replace=False)
                            
                            batch_t = {
                                k: torch.from_numpy(
                                    combined_stats["states"][k][batch_idx, :-1].reshape(-1, *self.obs_dims[k])
                                ).float().to(self.device)
                                for k in self.obs_dims
                            }

                            if self.q_mask:
                                raw_rewards = torch.from_numpy(combined_stats["rewards"][batch_idx]).float().to(self.device)
                                done = torch.from_numpy(combined_stats["done"][batch_idx]).float().to(self.device)
                                
                                episode_cum_rewards = torch.zeros_like(raw_rewards)
                                episode_cum_rewards[:, 0] = raw_rewards[:, 0]
                                for t in range(1, raw_rewards.shape[1]):
                                    episode_cum_rewards[:, t] = (1 - done[:, t-1]) * episode_cum_rewards[:, t-1] + raw_rewards[:, t]
                                complete_mask = (episode_cum_rewards >= self.best_reward_threshold_for_success) & (raw_rewards == 0)
                                mask = ~complete_mask
                                mask = mask.reshape(-1)
                                batch_t = {k: batch_t[k][mask] for k in self.obs_dims}
                            
                            # update actor
                            B_fm = next(iter(batch_t.values())).shape[0]
                            chunk_fm = B_fm // self.mini_batch_split
                            metrics = None
                            self.fm_optimizer.zero_grad(set_to_none=True)
                            for _i in range(self.mini_batch_split):
                                _sl = slice(_i * chunk_fm, (_i + 1) * chunk_fm if _i < self.mini_batch_split - 1 else B_fm)
                                _chunk = {k: batch_t[k][_sl] for k in batch_t}
                                
                                # RLDT | QAM | FQL | SDAC — swap the line below to change algorithm
                                _m = self._update_actor(fm_iteration, _chunk, self.mini_batch_split)
                                if metrics is None:
                                    metrics = {k: v / self.mini_batch_split for k, v in _m.items()}
                                else:
                                    for k in metrics:
                                        metrics[k] += _m[k] / self.mini_batch_split
                            self.fm_optimizer.step()
                            self.fm_lr_scheduler.step()
                            
                            total_actor_loss.append(metrics["actor_loss"])
                            total_kl_loss.append(metrics["kl_loss"])
                            grad_q_norm.append(metrics["q_norm"])
                            total_consist_loss.append(metrics["consistency_loss"])
                            total_action_sample_dist.append(metrics["action_sample_distance"])
                            pbar.set_postfix({"actor_loss": f"{metrics['actor_loss']:.4f}", "kl_loss": f"{metrics['kl_loss']:.4f}", "lr": f"{self.fm_optimizer.param_groups[0]['lr']:.4f}"})
                            pbar.update(1)
                        pbar.close()

                training_end = timer() # time for one iteration training (including Q update and policy update)

                # save current log as chunk
                prefetcher.save_chunk(fresh_stats)
                
                # EMA update actor from actor_ft
                for target_param, source_param in zip(
                    self.model.actor.parameters(), self.model.actor_ft.parameters()
                ):
                    target_param.data.copy_(
                        self.actor_ema_mom * target_param.data + (1 - self.actor_ema_mom) * source_param.data
                    )
                self.model.actor.eval()
                for p in self.model.actor.parameters():
                    p.requires_grad_(False)
            
            else:
                training_end = timer()
                    
            # Save model: numbered snapshot at save_model_freq, latest-only every other iteration
            if (self.itr % self.save_model_freq == 0 and self.itr > 0) or self.itr == self.n_train_itr - 1:
                self.save_model(save_numbered=True)

            # save the latest checkpoint
            self.save_model()
            torch.cuda.empty_cache()

            fileio_t = timer()
            log.info("Time for rollout: {:.4f} sec | time for training: {:.4f} sec | time for file I/O: {:.4f} sec".format(
                rollout_end, training_end, fileio_t
            ))

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )  
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "eval/success rate": success_rate,
                                "eval/avg episode reward": avg_episode_reward,
                                "eval/avg best reward": avg_best_reward,
                                "eval/num episode": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    avg_q_loss_epoch = float(np.mean(total_q_loss)) if len(total_q_loss) > 0 else 0.0
                    avg_actor_loss_epoch = float(np.mean(total_actor_loss)) if len(total_actor_loss) > 0 else 0.0
                    kl_loss_epoch = float(np.mean(total_kl_loss)) if len(total_kl_loss) > 0 else 0.0
                    avg_grad_q_norm = float(np.mean(grad_q_norm)) if len(grad_q_norm) > 0 else 0.0
                    avg_consist_loss = float(np.mean(total_consist_loss)) if len(total_consist_loss) > 0 else 0.0
                    avg_action_sample_dist = float(np.mean(total_action_sample_dist)) if len(total_action_sample_dist) > 0 else 0.0
                    avg_rollout_qvals = float(np.mean(qvals)) if len(qvals) > 0 else 0.0
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | q loss {avg_q_loss_epoch:8.4f} |  actor_loss {avg_actor_loss_epoch:8.4f} | reward {avg_episode_reward:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "train/avg episode reward": avg_episode_reward,
                                "train/num episode": num_episode_finished,
                                "train/success rate": success_rate,
                                "train/avg q loss": avg_q_loss_epoch,
                                "train/avg actor loss": avg_actor_loss_epoch,
                                "train/avg kl loss": kl_loss_epoch,
                                "train/avg action sample dist": avg_action_sample_dist,
                                "train/avg q grad norm": avg_grad_q_norm,
                                "train/avg consistency loss": avg_consist_loss,
                                "train/avg rollout q": avg_rollout_qvals,
                                "Q network lr": self.q_optimizer.param_groups[0]["lr"],
                                "Flow matching policy lr": self.fm_optimizer.param_groups[0]["lr"],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
    
    @torch.no_grad()    
    def _train_sampling(self, cond):
        """ Exploration Code for sampling actions during training """
        # normal sampling
        if self.sample_mode == "full":
            samples, _ = self.model.cache_sampling(cond=cond, deterministic=False, forward_net=self.model.actor_ft)
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
        else:
            raise NotImplementedError(f"{self.sample_mode} not implemented")
            
        # gaussian noisy exploration
        return samples

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
            
            # with open(os.path.join(self.checkpoint_dir, "buffer_{self.itr}.pkl"), "wb") as f:
            #     pickle.dump(self.old_stats, f, protocol=4)
            # np.save(os.path.join(self.checkpoint_dir, f"buffer_{self.itr}.npy"), self.old_stats)
        else:
            # always overwrite the rolling "latest" files for crash recovery
            for fname in ("state_latest.pt", "buffer_latest.pkl"):
                fpath = os.path.join(self.checkpoint_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
            
            torch.save(data, os.path.join(self.checkpoint_dir, "state_latest.pt"))
            # np.save(os.path.join(self.checkpoint_dir, "buffer_latest.npy"), self.old_stats)
            # with open(os.path.join(self.checkpoint_dir, "buffer_latest.pkl"), "wb") as f:
            #     pickle.dump(self.old_stats, f, protocol=4)
            
            log.info(f"Saved latest checkpoint at iteration {self.itr}")
            
    def load_model(self, load_path):
        # Prefer the rolling "latest" checkpoint saved every iteration
        latest_model_path = os.path.join(load_path, "state_latest.pt")
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

class QAdjImgAgent(QImgAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.q_mask, "QAdjImgAgent requires q_mask to be True for reward shaping"
        
        # Q Adj configuration
        self.inv_temp = cfg.train.inv_temp
        self.kl_coef = cfg.train.kl_coef
    
    @torch.no_grad()
    def _compute_adjoint_trajectories(self, cond):
        """
        Forward SDE (actor_ft for steps 0..T-2, actor for last step) then
        backpropagate the Q-gradient adjoint through actor (slow) dynamics.

        Args:
            cond: dict of obs tensors, each (B, *obs_shape), on device
        Returns:
            xs:        (T, B, H, A)  – trajectory at each ODE step
            adjs:      (T, B, H, A)  – adjoint at each ODE step
            ts:        (T, B)        – time values
            x_T:       (B, H, A)    – final action
            feat_slow: (B, feat_dim) – pre-encoded slow-actor features (reused in loss)
            info:      dict with adj statistics
        """
        B = cond["state"].shape[0]
        ode_steps = self.model.ode_steps
        h = 1.0 / ode_steps

        flat_cond = _flat_cond(cond, self.obs_dims)
        feat_ft   = self.model.actor_ft.encode_rgb(flat_cond)  # (B, feat_dim)
        feat_slow = self.model.actor.encode_rgb(flat_cond)     # (B, feat_dim)

        x = torch.randn(B, self.act_steps, self.cfg.action_dim, device=self.device)
        xs, ts_list = [], []

        for i in range(ode_steps):
            t_scalar = i / ode_steps
            t = torch.full((B,), t_scalar, device=self.device)
            xs.append(x.clone())
            ts_list.append(t)

            if i < ode_steps - 1:
                v = self.model.actor_ft.predict_velocity(feat_ft, x, t)
                sigma = math.sqrt(2.0 * (1 - t_scalar + h) / (t_scalar + h))
                x = x + h * (2 * v - x / (t_scalar + h)) + math.sqrt(h) * sigma * torch.randn_like(x)
            else:
                v = self.model.actor.predict_velocity(feat_slow, x, t)
                x = x + h * v

        x_T = x  # (B, H, A)

        # Adjoint init: p_T = -inv_temp * dQ/dx_T
        with torch.enable_grad():
            x_T_req  = x_T.clone().requires_grad_(True)
            x_T_flat = x_T_req.reshape(B, -1)
            q_val, _ = self.q_net(flat_cond, x_T_flat, reduction="mean")
            adj = -torch.autograd.grad(q_val.sum(), x_T_req)[0].detach() * self.inv_temp

        info = {"adj_max": adj.abs().max().item(), "adj_mean": adj.abs().mean().item()}

        # Adjoint backward: p_i = p_{i+1} + h * (df/dx_i)^T p_{i+1}
        # where f(x) = 2*actor_slow(x, t+h) - x/(t+h)
        adjs = [None] * ode_steps
        for i in reversed(range(ode_steps)):
            t_scalar = i / ode_steps
            t_ph = torch.full((B,), t_scalar + h, device=self.device)
            with torch.enable_grad():
                x_i    = xs[i].detach().requires_grad_(True)
                v_slow = self.model.actor.predict_velocity(feat_slow.detach(), x_i, t_ph)
                fn_val = 2 * v_slow - x_i / (t_scalar + h)
                vjp    = torch.autograd.grad(fn_val, x_i, grad_outputs=adj, create_graph=False)[0]
            adj     = (adj + h * vjp).detach()
            adjs[i] = adj

        xs_t   = torch.stack(xs,      dim=0)  # (T, B, H, A)
        adjs_t = torch.stack(adjs,    dim=0)  # (T, B, H, A)
        ts_t   = torch.stack(ts_list, dim=0)  # (T, B)
        return xs_t, adjs_t, ts_t, x_T, feat_slow, info

    def update_flow_with_adjoint_matching(self, batch_t, loss_scale=1.0):
        """
        One policy-update step using QAM adjoint matching for image observations.
        Loss = adjoint_matching_loss + kl_coef * kl_regularisation
        Caller owns zero_grad / optimizer.step / lr_scheduler.step.
        """
        cond = {k: batch_t[k] for k in self.obs_dims if k in batch_t}
        B = cond["state"].shape[0]
        ode_steps = self.model.ode_steps
        h = 1.0 / ode_steps

        xs, adjs, ts, x_T, feat_slow, adj_info = self._compute_adjoint_trajectories(cond)
        T, _, H, A = xs.shape

        # sigma_t = sqrt(2 * (1 - t + h) / (t + h))
        t_vals = torch.arange(ode_steps, dtype=torch.float32, device=self.device) / ode_steps
        sigmas = torch.sqrt(2.0 * (1 - t_vals + h) / (t_vals + h)).view(T, 1, 1, 1)

        xs_flat   = xs.view(T * B, H, A)
        ts_flat   = ts.view(T * B)
        flat_cond = _flat_cond(cond, self.obs_dims)

        with torch.enable_grad():
            # Re-encode actor_ft inside the compute graph (backprop through backbone)
            encoded_feat_ft = self.model.actor_ft.encode_rgb(flat_cond)         # (B, feat_dim)
            feat_ft_exp     = encoded_feat_ft.unsqueeze(0).expand(T, -1, -1).reshape(T * B, -1)

            vf_fine = self.model.actor_ft.predict_velocity(feat_ft_exp, xs_flat, ts_flat).view(T, B, H, A)

            with torch.no_grad():
                feat_slow_exp = feat_slow.detach().unsqueeze(0).expand(T, -1, -1).reshape(T * B, -1)
                vf_base = self.model.actor.predict_velocity(feat_slow_exp, xs_flat, ts_flat).view(T, B, H, A)

            # Adjoint matching loss: ||(vf_fine - vf_base)*(2/sigma) + sigma*adj||^2
            residual = (vf_fine - vf_base) * (2.0 / sigmas) + sigmas * adjs
            adj_loss = residual.pow(2).sum(dim=(-1, -2)).sum(dim=0).mean()

            # KL regularisation: keep actor_ft close to actor on random interpolations
            old_t      = torch.rand(B, device=self.device)
            old_t_view = old_t.view(B, 1, 1)
            noise_act  = torch.randn(B, H, A, device=self.device)
            old_x_t    = (1 - old_t_view) * noise_act + old_t_view * x_T.detach()
            with torch.no_grad():
                old_target    = self.model.actor.predict_velocity(feat_slow.detach(), old_x_t, old_t)
            old_flow_pred = self.model.actor_ft.predict_velocity(encoded_feat_ft, old_x_t, old_t)
            kl_loss = torch.mean(torch.sum((old_flow_pred - old_target) ** 2, dim=(-1, -2))) * self.kl_coef

            total_loss = adj_loss + kl_loss
            (total_loss / loss_scale).backward()

        with torch.no_grad():
            x_T_flat = x_T.reshape(B, -1)
            action_sample_distance = (
                torch.cdist(x_T_flat.unsqueeze(0), x_T_flat.unsqueeze(0)).mean().item() if B > 1 else 0.0
            )

        return {
            "actor_loss":           adj_loss.item(),
            "kl_loss":                kl_loss.item(),
            "q_norm":                 adj_info["adj_mean"],
            "consistency_loss":       0.0,
            "action_sample_distance": action_sample_distance,
        }
        
    def _update_actor(self, step_idx, batch_t, loss_scale=1.0):
        """ Update actor with QAM adjoint matching loss. Caller owns zero_grad / optimizer.step / lr_scheduler.step. """
        return self.update_flow_with_adjoint_matching(batch_t, loss_scale)