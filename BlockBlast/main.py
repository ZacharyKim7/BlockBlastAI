import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import BlockBlast
from BlockBlast.wrappers import ObservationMode, TerminateIllegalBlockBlast, RewardMode, AddStraight
from tqdm import tqdm


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "BlockBlastAI"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BlockBlast/BlockBlast-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # track interval
    log_charts_interval: int = 2000
    """Record interval for chart"""
    log_losses_interval: int = 100
    """Record interval for losses"""
    record_interval: int = 20000
    """Record interval for RecordVideo"""

    load_model: str = ""
    """whether to load model `runs/{run_name}` folder"""

    # game option
    n_channel: int = 4
    """number of channel"""

    # reward


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", crash33=False)
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: (x % args.record_interval == 0)
            )
        else:
            env = gym.make(env_id, crash33=False)

        env = RewardMode(env, "BlockBlast")
        env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
        env = ObservationMode(env, n_channel=args.n_channel)
        env = AddStraight(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(args.n_channel, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.network2 = nn.Sequential(
            layer_init(nn.Linear(64 * 5 * 5 + 1, 1024)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(1024, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(1024, 1), std=1)

    def pre_get(self, x, straight):
        x = self.network(x)
        x = torch.concat([x, straight / 10], dim=-1)
        x = self.network2(x)
        return x

    def get_value(self, x, straight):
        return self.critic(self.pre_get(x, straight))

    def get_action_and_value(self, x, straight, action=None, invalid_action_mask=None):
        hidden = self.pre_get(x, straight)
        logits = self.actor(hidden)
        # probs = Categorical(logits=logits)
        probs = CategoricalMasked(logits=logits, masks=invalid_action_mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
