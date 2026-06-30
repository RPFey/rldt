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


class QAMAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
    
        # initialize the optimizer
        self.fm_policy = self.model.actor_ft  # Point to the base flow matching policy
        assert self.model.prediction_type == 'v_pred', "QAM currently only supports v_prediction type for the flow matching policy"
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
        self.q_update_steps = cfg.train.q_update_steps
        self.q_split_mode = cfg.train.q_split_mode
        self.weighted_sample = cfg.train.weighted_sample
        self.policy_update_steps = cfg.train.policy_update_steps
        self.gradq_net_idx = cfg.train.gradq_net_idx
        self.q_bootstrap = cfg.train.q_bootstrap
        self.td_lambda = cfg.train.td_lambda
        self.q_mask = cfg.train.q_mask

        # QAM adjoint matching configuration
        self.inv_temp = cfg.train.inv_temp
        self.clip_adj = cfg.train.clip_adj
        self.kl_coef = cfg.train.kl_coef
        self.policy_ema_tau = cfg.train.policy_ema_tau                       # EMA rate for actor ← actor_ft update
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        self.q_target_max, self.q_target_min = cfg.train.q_target_max, cfg.train.q_target_min
        self.rho = cfg.train.rho
        self.new_fraction = cfg.train.new_fraction
        # repeat the target
        self.q_repeat = cfg.train.q_repeat
        
        # exploration related
        self.action_sample_noise = cfg.train.action_sample_noise
        self.sample_mode = cfg.train.sample_mode # "extrapolate"
        self.noise_mode = cfg.train.noise_mode  # "add"
        self.epsilon_ratio = cfg.train.epsilon_ratio  # 0.2
        self.min_extrp_step = cfg.train.min_extrp_step # 0
        
        self.q_batch_size = self.batch_size
        self.critic_initialized = False

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
    def _compute_adjoint_trajectories(self, obs, obs_critic=None):
        """
        Run forward SDE (actor_ft for steps 0..T-2, actor for last step) then
        backpropagate the Q-gradient adjoint through actor (slow) dynamics.

        Args:
            obs: (B, obs_dim)
            obs_critic: (B, obs_dim+1) time-augmented state for Q; defaults to obs
        Returns:
            xs:    (T, B, H, A)  – trajectory states at each ODE step
            adjs:  (T, B, H, A)  – adjoint states at each ODE step
            ts:    (T, B)        – time values at each ODE step
            x_T:   (B, H, A)    – final action after all steps
            info:  dict with adj statistics
        """
        B = obs.shape[0]
        ode_steps = self.model.ode_steps
        h = 1.0 / ode_steps

        x = torch.randn(B, self.act_steps, self.cfg.action_dim, device=self.device)
        xs, ts_list = [], []

        # Forward: SDE via actor_ft for all steps except the last (ODE via actor)
        for i in range(ode_steps):
            t_scalar = i / ode_steps
            t = torch.full((B,), t_scalar, device=self.device)
            xs.append(x.clone())
            ts_list.append(t)

            if i < ode_steps - 1:
                v = self.model.actor_ft(x, t, cond={"state": obs})
                sigma = math.sqrt(2.0 * (1 - t_scalar + h) / (t_scalar + h))
                x = x + h * (2 * v - x / (t_scalar + h)) + math.sqrt(h) * sigma * torch.randn_like(x)
            else:
                # last step: clean ODE with slow actor so x_T is differentiable w.r.t. Q
                v = self.model.actor(x, t, cond={"state": obs})
                x = x + h * v

        x_T = x  # (B, H, A)

        obs_q = obs if obs_critic is None else obs_critic

        # Adjoint initialisation: p_T = -dQ/dx_T * inv_temp
        with torch.enable_grad():
            x_T_req = x_T.clone().requires_grad_(True)
            x_T_flat = x_T_req.reshape(B, -1)
            if self.clip_adj:
                x_T_flat = x_T_flat.clamp(-1, 1)
            q_val, _ = self.q_net(obs_q, x_T_flat, reduction="mean")
            adj = -torch.autograd.grad(q_val.sum(), x_T_req)[0].detach() * self.inv_temp

        info = {"adj_max": adj.abs().max().item(), "adj_mean": adj.abs().mean().item()}

        # Adjoint backprop through slow actor:  p_i = p_{i+1} + h * (df/dx_i)^T p_{i+1}
        # where  f(x) = 2*actor(x, t+h) - x/(t+h)
        adjs = [None] * ode_steps
        for i in reversed(range(ode_steps)):
            t_scalar = i / ode_steps
            t_ph = torch.full((B,), t_scalar + h, device=self.device)
            with torch.enable_grad():
                x_i = xs[i].detach().requires_grad_(True)
                v_slow = self.model.actor(x_i, t_ph, cond={"state": obs})
                fn_val = 2 * v_slow - x_i / (t_scalar + h)
                vjp = torch.autograd.grad(fn_val, x_i, grad_outputs=adj, create_graph=False)[0]
            adj = (adj + h * vjp).detach()
            adjs[i] = adj

        xs_t   = torch.stack(xs,   dim=0)  # (T, B, H, A)
        adjs_t = torch.stack(adjs, dim=0)  # (T, B, H, A)
        ts_t   = torch.stack(ts_list, dim=0)  # (T, B)
        return xs_t, adjs_t, ts_t, x_T, info

    def update_flow_with_adjoint_matching(self, batch):
        """
        One policy update step using QAM adjoint matching.

        Loss = adjoint_matching_loss + kl_coef * kl_regularisation
        """
        state = batch["state"].to(self.device)
        batch_size = state.shape[0]
        ode_steps = self.model.ode_steps
        h = 1.0 / ode_steps

        if self.rollout_stage_cond and "timestep" in batch:
            state_critic = torch.cat([state, batch["timestep"].to(self.device)], dim=-1)
        else:
            state_critic = state

        # Expand states for n_particles
        BP = state.shape[0]

        # Compute adjoint trajectories (no gradient accumulation in actor parameters)
        xs, adjs, ts, x_T, adj_info = self._compute_adjoint_trajectories(state, obs_critic=state_critic)
        # xs, adjs: (T, BP, H, A);  ts: (T, BP)
        T, _, H, A = xs.shape

        # sigma_t = sqrt(2 * (1 - t + h) / (t + h))  – same as used in the SDE forward pass
        t_vals = torch.arange(ode_steps, dtype=torch.float32, device=self.device) / ode_steps
        sigmas = torch.sqrt(2.0 * (1 - t_vals + h) / (t_vals + h)).view(T, 1, 1, 1)

        # Flatten time and batch dimensions for a single batched actor call
        xs_flat    = xs.view(T * BP, H, A)
        ts_flat    = ts.view(T * BP)
        state_flat = state.unsqueeze(0).expand(T, -1, -1).reshape(T * BP, -1)

        self.fm_optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            vf_fine = self.model.actor_ft(xs_flat, ts_flat, cond={"state": state_flat}).view(T, BP, H, A)

            with torch.no_grad():
                vf_base = self.model.actor(xs_flat, ts_flat, cond={"state": state_flat}).view(T, BP, H, A)

            # Adjoint matching loss:  ||(vf_fine - vf_base) * 2/sigma + sigma * adj||^2
            # sum over (H, A), then sum over T, then mean over B — matches official implementation
            residual  = (vf_fine - vf_base) * (2.0 / sigmas) + sigmas * adjs
            adj_loss  = residual.pow(2).sum(dim=(-1, -2)).sum(dim=0).mean()

            # KL regularisation: keep actor_ft close to actor on random interpolations
            old_t      = torch.rand(BP, device=self.device)
            old_t_view = old_t.view(BP, 1, 1)
            noise_act  = torch.randn(BP, H, A, device=self.device)
            old_x_t    = (1 - old_t_view) * noise_act + old_t_view * x_T.detach()
            with torch.no_grad():
                old_target = self.model.actor(old_x_t, old_t, cond={"state": state})
            old_flow_pred = self.model.actor_ft(old_x_t, old_t, cond={"state": state})
            kl_loss = torch.mean(torch.sum((old_flow_pred - old_target) ** 2, dim=(-1, -2))) * self.kl_coef

            total_loss = adj_loss + kl_loss
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.fm_policy.parameters(), max_norm=1.0)
        self.fm_optimizer.step()
        self.fm_lr_scheduler.step()

        return {
            "fm_adj_loss": adj_loss.item(),
            "kl_loss":     kl_loss.item(),
            **adj_info,
        }
    
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
    def update_q_net_td_lambda(self, fresh_stat, td_lambda = 0.9, update_steps=10, num_sample_actions=4):
        # compute the target
        states = torch.from_numpy(fresh_stat["states"]).to(self.device).float()
        actions = torch.from_numpy(fresh_stat["actions"]).to(self.device).float()
        n_envs, time_step, action_dim = fresh_stat["actions"].shape
        _, _, obs_dim = fresh_stat["states"].shape
        raw_rewards = torch.from_numpy(fresh_stat["rewards"]).to(self.device).float()
        done = torch.from_numpy(fresh_stat["done"]).to(self.device).float()

        if self.rollout_stage_cond:
            timesteps = torch.from_numpy(fresh_stat["timesteps"]).to(self.device).float()
            states_critic = torch.cat([states, timesteps.unsqueeze(-1)], dim=-1)
        else:
            states_critic = states

        # compute the mask using raw rewards (q_target_max threshold is in raw reward space)
        if self.q_mask:
            cum_rewards = torch.cumsum(raw_rewards, dim=1)  # (n_envs, time_step)
            complete_mask = (cum_rewards >= self.best_reward_threshold_for_success) & (raw_rewards == 0) # for complete dangling cases
            mask = ~complete_mask                                               # mask out the states after reaching the max reward, to avoid learning from dangling states
            done = ((done > 0.) | (cum_rewards >= self.best_reward_threshold_for_success)).float()   # treat the states after reaching max reward as done states, to avoid learning from dangling states
        else:
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
                _q_mean, _q_std = self.q_net.compute_target(states_critic, policy_actions, reduction="mean")
                _q_target = (_q_mean - self.rho * _q_std).reshape(n_envs, time_step)
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
                        samples = self.model(cond=cond, deterministic=True, forward_net=self.model.actor_ft)
                    else:
                        samples = self._train_sampling(cond)
                        
                    actions = samples.trajectories[:, :self.act_steps, :].reshape(self.n_envs, -1)
                    if self.rollout_stage_cond:
                        r_feat = torch.from_numpy(episode_returns).to(self.device).reshape(-1, 1).float()
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
                fresh_stats["timesteps"][:, step + 1] = episode_returns

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

            # # Reset Q network
            # if self.itr % self.cfg.train.buffer_num_episode == 1:
            #     for q_net in self.q_net.q_nets:
            #         self.q_net._init_module(q_net)
            #     self.critic_initialized = False

            # update model
            total_q_loss, total_adj_loss, total_kl_loss = [], [], []
            q_improves, adj_means = [], []
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
                            nonzero_trajs_idx = np.nonzero(np.sum(np.abs(self.old_stats["rewards"]), axis=-1))[0]
                            iszero_traj_idx = np.where(np.sum(np.abs(self.old_stats["rewards"]), axis=-1) < 1e-3)[0]
                            # sample half-half with nonzero & zero trajs
                            old_nonzero_idx = np.random.choice(
                                nonzero_trajs_idx, 
                                size = old_nums // 2, 
                                replace = True if len(nonzero_trajs_idx) < old_nums // 2 else False
                            )
                            old_zero_idx = np.random.choice(
                                iszero_traj_idx, 
                                size = old_nums // 2, 
                                replace = True if len(iszero_traj_idx) < old_nums // 2 else False
                            )
                            old_idx = np.concatenate([old_nonzero_idx, old_zero_idx], axis=0)
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
                    cur_q_update_steps = self.q_update_steps // total_iterations if self.critic_initialized else 1024 # warm up with more steps
                    # TODO: Full td lambda range from 0 - 1.
                    td_lambda = self.td_lambda if self.td_lambda < 1.0 else max( min( 1 - i / (total_iterations - 1), 1) , 0) # 1 - 0
                    q_error = self.update_q_net_td_lambda(combined_stats, td_lambda=td_lambda, update_steps=cur_q_update_steps, num_sample_actions=self.q_bootstrap)
                    total_q_loss.append(q_error)
                    self.critic_initialized = True
                    
                    # gradq_net_idx > 0; alternating between ensembles
                    # otherwise, -1 will default to average
                    self.gradq_net_idx = self.itr % len(self.q_net.q_nets) if self.gradq_net_idx >= 0 else -1 

                    # update flow matching policy with SVGD
                    if self.itr > self.n_critic_warmup_itr:
                        fm_batch_size = combined_stats["states"].shape[0] // 8
                        fm_update_steps = self.policy_update_steps // total_iterations
                        pbar = tqdm(range(fm_update_steps), desc=f"FM update itr {self.itr}")
                        for _ in range(fm_update_steps):
                            self._check_cm()
                            
                            # Sample index
                            batch_idx = np.random.choice(
                                combined_stats["states"].shape[0], size=fm_batch_size, replace=False)
                            
                            batch_t = {
                                # filter out the last state and action for each traj
                                "state": torch.from_numpy(
                                    combined_stats["states"][batch_idx, :-1, :].reshape(-1, self.cfg.obs_dim)
                                ).float().to(self.device)
                            }
                            if self.rollout_stage_cond:
                                batch_t["timestep"] = torch.from_numpy(
                                    combined_stats["timesteps"][batch_idx, :-1].reshape(-1, 1)
                                ).float().to(self.device)

                            if self.q_mask:
                                rewards = torch.from_numpy(    
                                    combined_stats["rewards"][batch_idx, :]
                                ).float().to(self.device)
                                complete_mask = (torch.cumsum(rewards, dim=1) >= self.q_target_max) & (rewards == 0) # for complete dangling cases
                                mask = ~complete_mask
                                mask = mask.reshape(-1)
                                batch_t["state"] = batch_t["state"][mask]
                                
                            metrics = self.update_flow_with_adjoint_matching(batch_t)
                            total_adj_loss.append(metrics["fm_adj_loss"])
                            total_kl_loss.append(metrics["kl_loss"])
                            adj_means.append(metrics["adj_mean"])
                            pbar.set_postfix({"adj_loss": f"{metrics['fm_adj_loss']:.4f}", "kl_loss": f"{metrics['kl_loss']:.4f}", "lr": f"{self.fm_optimizer.param_groups[0]['lr']:.4f}"})
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

                # EMA update: actor ← tau * actor + (1 - tau) * actor_ft
                with torch.no_grad():
                    for p_slow, p_fast in zip(self.model.actor.parameters(), self.model.actor_ft.parameters()):
                        p_slow.data.mul_(self.policy_ema_tau).add_(p_fast.data, alpha=1 - self.policy_ema_tau)
                    
            # Save model: numbered snapshot at save_model_freq, latest-only every other iteration
            if (self.itr % self.save_model_freq == 0 and self.itr > 0) or self.itr == self.n_train_itr - 1:
                self.save_model(save_numbered=True)

            # save the latest checkpoint
            self.save_model()

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
                    avg_q_loss_epoch  = float(np.mean(total_q_loss))   if total_q_loss   else 0.0
                    avg_adj_loss_epoch = float(np.mean(total_adj_loss)) if total_adj_loss  else 0.0
                    avg_q_improve      = float(np.mean(q_improves))     if q_improves      else 0.0
                    kl_loss_epoch      = float(np.mean(total_kl_loss))  if total_kl_loss   else 0.0
                    avg_adj_mean       = float(np.mean(adj_means))      if adj_means       else 0.0
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | q loss {avg_q_loss_epoch:8.4f} | adj_loss {avg_adj_loss_epoch:8.4f} | reward {avg_episode_reward:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "train/avg episode reward": avg_episode_reward,
                                "train/num episode": num_episode_finished,
                                "train/success rate": success_rate,
                                "train/avg q loss": avg_q_loss_epoch,
                                "train/avg adj loss": avg_adj_loss_epoch,
                                "train/avg kl loss": kl_loss_epoch,
                                "train/q improvement": avg_q_improve,
                                "train/avg adj mean": avg_adj_mean,
                                "train/inv_temp": self.inv_temp,
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

    def _compute_q(self, batch, num_sample_actions=2):
        """ Debugging functions to compute Q values on a batch of data """
        # compute the target
        states = torch.from_numpy(batch["states"]).to(self.device).float()
        n_envs, time_step, obs_dim = states.shape
        
        # compute the policy actions
        policy_actions_repeat = []
        for _ in range(num_sample_actions):
            policy_actions = self.model({"state": states.reshape(-1, obs_dim)}, deterministic=False, forward_net=self.model.actor_ft)
            policy_actions = policy_actions.trajectories[:, :self.act_steps, :].reshape(n_envs, time_step, -1)
            policy_actions_repeat.append(policy_actions)
        
        # Get target Q values using the current Q network (no grad for target computation)
        q_val = torch.zeros_like(states[..., 0])
        for policy_actions in policy_actions_repeat:
            _q_target, _ = self.q_net(states, policy_actions, reduction="mean")
            _q_target = _q_target.reshape(n_envs, time_step)
            q_val += _q_target / num_sample_actions

        logging.info(f"Evaluated Q values with avg: {q_val.mean().item():.4f}, std: {q_val.std().item():.4f}")
        return q_val.mean().item()

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

        self.critic_initialized = True

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
        
