"""
Policy gradient with diffusion policy. VPG: vanilla policy gradient

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

import copy
import torch
import logging

log = logging.getLogger(__name__)
import torch.nn.functional as F

from model.diffusion.flowmatching import FlowMatchingModel
from model.diffusion.sampling import make_timesteps, extract
from torch.distributions import Normal


class FM_SVGD(FlowMatchingModel):
    """ Use SVGD to finetune Flow Matching """
    def __init__(
        self,
        actor,
        critic,
        network_path=None,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )

        # Re-name network to actor
        self.actor_ft = self.network

        # Make a copy of the original model
        self.actor = copy.deepcopy(self.actor_ft)
        logging.info("Cloned model for fine-tuning")

        # Turn off gradients for original model
        for param in self.actor.parameters():
            param.requires_grad = False
        logging.info("Turned off gradients of the pretrained network")
        
        logging.info(
            f"Number of finetuned parameters: {sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)}"
        )

        # Value function
        self.critic = critic.to(self.device)

