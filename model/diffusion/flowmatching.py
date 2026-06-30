import torch
import logging
from torch import nn
import torch.nn.functional as F
from collections import namedtuple

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class FlowMatchingModel(nn.Module):
    """
    Flow Matching policy model with the same *essential* API as DiffusionModel.
    """

    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        device="cuda:0",
        denoising_steps=50,
        final_action_clip_value=None,
        network_path=None,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ode_steps = denoising_steps
        self.final_action_clip_value = final_action_clip_value
        self.prediction_type = kwargs.get("prediction_type", "v_pred")
        assert self.prediction_type in ["v_pred", "x_pred"], "prediction_type should be in v_pred, x_pred"

        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded RL-trained policy from %s", network_path)
        
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

    # ------------------------------------------------------------
    # Sampling (ODE integration)
    # ------------------------------------------------------------
    @torch.no_grad()
    def forward(self, cond, 
                deterministic=False, 
                forward_net:nn.Module = None):
        """
        Deterministic sampling via ODE integration.

        Args:
            cond: dict with keys like "state" or "rgb"
        Returns:
            Sample(trajectories, None)
        """
        device = self.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]    
        
        B = len(sample_data)
        
        # assign velocity field
        forward_net = self.network if forward_net is None else forward_net

        # Initial noise x(0) ~ N(0, I)
        if deterministic:
            x = torch.zeros(
                (B, self.horizon_steps, self.action_dim), device=device
            )
        else:
            x = torch.randn(
                (B, self.horizon_steps, self.action_dim), device=device
            )

        dt = 1.0 / self.ode_steps
        trajs = [x]

        for i in range(self.ode_steps):
            t = torch.full((B,), i / self.ode_steps, device=device)

            if self.prediction_type == 'v_pred':
                # Velocity field v_theta(x, t | cond) and Euler Integration
                v = forward_net(x, t, cond=cond)
                x = x + dt * v
            elif self.prediction_type == 'x_pred':
                # Compute velocity and Euler Integration
                x_target = forward_net(x, t, cond = cond)
                x = x + dt * (x_target - x) / (1 - t.view(B, 1, 1))
            
            trajs.append(x)

        if self.final_action_clip_value is not None:
            x = torch.clamp(
                x,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        trajs = torch.stack(trajs, dim=1)  # (B, T, horizon_steps, action_dim)
        return Sample(x, trajs)
    
    @torch.no_grad()
    def train_sampling(self, 
                       cond, 
                       stop_idx:int = None,
                       deterministic: bool = True,
                       random_noise: float= 0.01,
                       forward_net:nn.Module = None):
        """ Training time sampling, stop-idx is the timestep that the remaining part is exptrapolated by the current velocity"""
        device = self.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        
        # assign velocity field
        forward_net = self.network if forward_net is None else forward_net
        final_generation = torch.zeros((B, self.horizon_steps, self.action_dim), device=device)
        
        # Initial noise x(0) ~ N(0, I)
        x = torch.randn(
            (B, self.horizon_steps, self.action_dim), device=device
        )

        dt = 1.0 / self.ode_steps
        for i in range(self.ode_steps):
            t = torch.full((B,), i / self.ode_steps, device=device)

            # Velocity field v_theta(x, t | cond)
            pred = forward_net(x, t, cond=cond)
            
            # Euler integration
            if self.prediction_type == 'v_pred':
                E_x = x + pred * (1.0 - t.view(B, 1, 1))
                x = x + dt * pred
            elif self.prediction_type == 'x_pred':
                E_x = pred
                x = x + dt * (pred - x) / (1 - t.view(B, 1, 1))
                
            if not deterministic:
                noise = torch.randn_like(x)
                E_x = E_x + ( (1.0 - t.view(B, 1, 1)) ** 0.5 ) * (random_noise * noise)
                x = x # + (dt ** 0.5) * (random_noise * noise)                
            
            # estimate the final target
            final_generation[stop_idx == i] = E_x[stop_idx == i] 
            
        if self.final_action_clip_value is not None:
            final_generation = torch.clamp(
                final_generation,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return Sample(final_generation, None)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def loss(self, x, cond):
        """
        Entry point to match DiffusionModel.loss()
        """
        return self.fm_loss(x, cond)

    def fm_loss(self, x_start, cond):
        """
        Standard Flow Matching loss:

        x_t = (1 - t) * z + t * x_0
        v*  = x_0 - z

        L = E[ || v_theta(x_t, t) - v* ||^2 ]
        """
        device = x_start.device
        B = x_start.shape[0]

        # Sample base noise
        z = torch.randn_like(x_start)

        # Sample continuous time t ~ U[0, 1]
        t = torch.rand(B, device=device)

        # Interpolate
        t_view = t.view(B, 1, 1)
        x_t = (1.0 - t_view) * z + t_view * x_start

        if self.prediction_type == 'v_pred':
            # Ground-truth velocity
            v_target = x_start - z
            # Predict velocity
            v_pred = self.network(x_t, t, cond=cond)
            return F.mse_loss(v_pred, v_target, reduction="mean")
        
        elif self.prediction_type == 'x_pred':
            # Predict position
            x_pred = self.network(x_t, t, cond=cond)
            l2_norm = torch.mean( (x_pred.reshape(B, -1) - x_start.reshape(B, -1)) ** 2, dim=-1)
            
            t_clamp = torch.clip(t, min=0, max= 1 - 1e-3)   # clamp to stabilize loss weight
            weighting = 1 / (1 - t_clamp) ** 2              # weighted by 1 / [1 - 2]**2
            mse_loss = torch.mean(l2_norm * weighting)
            return mse_loss

    @torch.no_grad()
    def cache_sampling(self, cond,
                       deterministic=False, 
                       forward_net: nn.Module = None,
                       num_replica: int = 1):
        """
        Sampling identical to forward() but encodes RGB/state only once,
        then calls predict_velocity at each ODE step.

        Args:
            cond: dict with keys like "state" or "rgb"
            deterministic: if True, start from zeros instead of Gaussian noise
            forward_net: optional network override (must expose encode_rgb / predict_velocity)
            num_replica (int): number of particles for each sample
        Returns:
            Sample(trajectories, chains)
        """
        device = self.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]

        B = len(sample_data)
        net = self.network if forward_net is None else forward_net

        # Encode visual + state features once
        cond_encoded = net.encode_rgb(cond)

        # Initial noise x(0)
        if deterministic:
            x = torch.zeros((B*num_replica, self.horizon_steps, self.action_dim), device=device)
        else:
            x = torch.randn((B*num_replica, self.horizon_steps, self.action_dim), device=device)

        dt = 1.0 / self.ode_steps
        trajs = [x]

        for i in range(self.ode_steps):
            t = torch.full((B*num_replica,), i / self.ode_steps, device=device)

            cond_encoded_replica = cond_encoded.repeat_interleave(num_replica, dim=0)
            if self.prediction_type == "v_pred":
                v = net.predict_velocity(cond_encoded_replica, x, t)
                x = x + dt * v
            elif self.prediction_type == "x_pred":
                x_target = net.predict_velocity(cond_encoded_replica, x, t)
                x = x + dt * (x_target - x) / (1 - t.view(B, 1, 1))
            trajs.append(x)

        if self.final_action_clip_value is not None:
            x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

        trajs = torch.stack(trajs, dim=1)  # (B, T+1, horizon_steps, action_dim)
        return Sample(x, trajs), cond_encoded
    