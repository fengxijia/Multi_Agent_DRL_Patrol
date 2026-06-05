"""Proximal Policy Optimization (PPO) learner for the patrolling task.

The PPO objective, network and optimisation logic are unchanged from the
original three project copies; this is the single shared implementation. Only
checkpoint handling was hardened (see :meth:`PPO.load`).
"""

import os

import numpy as np
import torch

from marl_patrol.buffers.rollout_buffer import BATCH_FIELDS, RolloutBatch
from marl_patrol.models.patrol_net import ActorCriticPolicy


def convert_batch_to_tensor(batch, device):
    """Move a numpy :class:`RolloutBatch` onto ``device`` as float tensors."""
    assert isinstance(batch, RolloutBatch)
    tensor_dict = {}
    for k in BATCH_FIELDS:
        if k == "observations":
            ag, tg, cn, mk = tuple(getattr(batch, k))
            tensor_dict[k] = [
                torch.FloatTensor(ag).to(device),
                torch.FloatTensor(tg).to(device),
                torch.FloatTensor(cn).to(device),
                torch.FloatTensor(mk).to(device),
            ]
        else:
            tensor_dict[k] = torch.FloatTensor(getattr(batch, k)).to(device)
    return RolloutBatch(**tensor_dict)


class PPO:
    """PPO learner wrapping an :class:`ActorCriticPolicy`."""

    def __init__(self, cfgs, device, policy: ActorCriticPolicy, phase=0):
        self.device = device
        self.phase = phase
        self.policy = policy(cfgs).to(device)
        # getattr: pick the optimiser class by name from torch.optim
        optimizer = getattr(torch.optim, cfgs.get("optim", "Adam"))
        self.lr = cfgs.getfloat("learning_rate")
        self.optimizer = optimizer(self.policy.parameters(), lr=self.lr)
        self.mse_loss = torch.nn.MSELoss()
        self.grad_max = cfgs.getfloat("max_grad")
        self.n_epochs = cfgs.getint("n_epochs")
        self.entropy_coef = cfgs.getfloat("entropy_coef")
        self.value_coef = cfgs.getfloat("value_coef")
        self.surr_clip = cfgs.getfloat("surrogate_loss_clip")

    def get_weights(self):
        return {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "phase": self.phase,
        }

    def set_weights(self, temp_best, load_all=False):
        if load_all:
            self.policy.load_state_dict(temp_best["model_state_dict"])
            self.optimizer.load_state_dict(temp_best["optimizer_state_dict"])
            self.phase = temp_best["phase"]
        else:
            self.policy.load_state_dict(temp_best["model_state_dict"])

    def act(self, obs, greedy=False):
        self.policy.eval()
        act, act_log_prob, value = self.policy.act(obs, greedy)
        return act, act_log_prob, value

    def evaluate(self, obs, act):
        value, act_log_prob, act_entropy = self.policy.evaluate(obs, act)
        return value, act_log_prob, act_entropy

    def update(self, batch):
        self.policy.train()

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []

        batch = convert_batch_to_tensor(batch, self.device)

        for _ in range(self.n_epochs):
            values, act_log_prob, act_entropy = self.evaluate(batch.observations, batch.actions)

            advantages = batch.advantages  # (B,)

            # ratio = exp(log p_theta - log p_theta_old) = p_theta / p_theta_old
            ratio = torch.exp(act_log_prob - batch.old_log_probs)
            surr_loss_1 = ratio * advantages
            surr_loss_2 = torch.clamp(ratio, 1 - self.surr_clip, 1 + self.surr_clip) * advantages

            policy_loss = -torch.min(surr_loss_1, surr_loss_2).mean()

            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.surr_clip).float()).item()
            clip_fractions.append(clip_fraction)

            value_loss = self.mse_loss(values.squeeze(), batch.returns)
            value_losses.append(value_loss.item())

            entropy_loss = -torch.mean(act_entropy)
            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Approximate reverse KL for monitoring / early stopping.
            # http://joschu.net/blog/kl-approx.html
            with torch.no_grad():
                log_ratio = act_log_prob - batch.old_log_probs
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                approx_kl_divs.append(approx_kl_div)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_max)
            self.optimizer.step()

        self.policy.eval()

        return (
            np.mean(value_losses),
            np.mean(pg_losses),
            np.mean(entropy_losses),
            np.mean(clip_fractions),
            np.mean(approx_kl_divs),
        )

    def save(self, model_path, suffix=""):
        model_path = os.path.join(model_path, f"checkpoint_{suffix}.pth")
        model_path = model_path.replace("\\", "/")
        print(f"Saving checkpoint to {model_path}")
        torch.save(
            {
                "phase": self.phase,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_path,
        )

    def load(self, model_path, suffix="", load_all=False):
        model_path = os.path.join(model_path, f"checkpoint_{suffix}.pth")
        model_path = model_path.replace("\\", "/")
        print(f"Loading checkpoint from {model_path}")
        # map_location lets a GPU-trained checkpoint load on CPU (e.g. at test time).
        # weights_only=False keeps the optimizer state / phase int loadable on
        # torch>=2.6; older torch (<1.13) doesn't accept the kwarg, hence the fallback.
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_all:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # BUGFIX: ``phase`` is a plain int, not an nn.Module -- assign it
            # directly instead of calling ``.load_state_dict`` on it.
            self.phase = checkpoint["phase"]
