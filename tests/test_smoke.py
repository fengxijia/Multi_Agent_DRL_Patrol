"""Lightweight smoke tests: imports, network forward pass, batching, env step.

These run on CPU in seconds and do not require a trained model. The full
training/eval smoke run is exercised via the CLI (see README).
"""

import configparser
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_cfg(extra=None):
    """A minimal in-memory config sufficient to build the env and learner."""
    cfg = configparser.ConfigParser()
    section = {
        "dim_unify": "16",
        "repeat_times_inside_net": "2",
        "heads": "8",
        "dim_af": "4",
        "dim_cf": "4",
        "dim_tf": "3",
        "env_size0": "1.0",
        "env_size1": "1.0",
        "numAgents0": "2",
        "numAgents1": "2",
        "numTargets0": "4",
        "numTargets1": "4",
        "numNeighbors0": "1",
        "numNeighbors1": "3",
        "learning_rate": "1e-3",
        "gae_lambda": "1",
        "gamma": "0.99",
        "window_size": "10",
        "batch_size": "8",
        "max_episode_steps": "20",
        "max_grad": "2",
        "n_epochs": "2",
        "entropy_coef": "0.01",
        "value_coef": "0.5",
        "surrogate_loss_clip": "0.2",
    }
    if extra:
        section.update(extra)
    cfg.read_dict({"s": section})
    return cfg["s"]


def test_imports():
    import marl_patrol  # noqa: F401
    from marl_patrol.algorithms.ppo import PPO  # noqa: F401
    from marl_patrol.buffers.rollout_buffer import RolloutBuffer  # noqa: F401
    from marl_patrol.envs import EveryTimeStepPatrolEnv, OnNodePatrolEnv  # noqa: F401
    from marl_patrol.models.patrol_net import ActorCriticPolicy  # noqa: F401
    from marl_patrol.train import common, every_timestep, on_node  # noqa: F401


def test_network_forward():
    from marl_patrol.models.patrol_net import ActorCriticPolicy

    net = ActorCriticPolicy(_tiny_cfg())
    n_agents, n_targets, bs = 2, 4, 5
    obs = (
        torch.randn(bs, n_agents, 4),
        torch.randn(bs, n_targets, 3),
        torch.randn(bs, n_targets, 4),
        torch.tensor(bs * [[[True] * n_targets]]),
    )
    value, policy = net(obs)
    assert value.shape[0] == bs
    assert torch.isfinite(value).all()


def test_make_batch_covers_all_samples():
    from marl_patrol.buffers.rollout_buffer import RolloutBatch
    from marl_patrol.train.common import make_batch

    n, n_targets = 10, 4

    def fake_traj(n):
        obs = (
            np.zeros((n, 2, 4)),
            np.zeros((n, n_targets, 3)),
            np.zeros((n, n_targets, 4)),
            np.zeros((n, 1, n_targets)),
        )
        return RolloutBatch(obs, np.arange(n), np.zeros(n), np.zeros(n), np.zeros(n))

    batches = make_batch(4, [fake_traj(n)])
    # No dropped samples, no empty batch.
    total = sum(b.actions.shape[0] for b in batches)
    assert total == n
    assert all(b.actions.shape[0] > 0 for b in batches)
    # Every original index appears exactly once.
    seen = np.concatenate([b.actions for b in batches])
    assert sorted(seen.tolist()) == list(range(n))


def test_on_node_env_runs():
    from marl_patrol.envs.on_node import MAPatrolEnv

    env = MAPatrolEnv()
    env.params_from_cfg(_tiny_cfg())
    env._initEnv()
    env.reset(scattered=True)
    assert env.numAgents == 2
    obs = env.observe(1, relative=True)
    assert len(obs) == 4
    # fixed_map reuse keeps the same targets.
    coords = env.targetCoords.copy()
    env.reset(scattered=True, fixed_map=True)
    assert np.allclose(env.targetCoords, coords)


def test_every_timestep_env_step():
    from marl_patrol.envs.every_timestep import MAPatrolEnv

    env = MAPatrolEnv()
    env.params_from_cfg(_tiny_cfg())
    env._initEnv()
    env.reset(scattered=True)
    env.communicate(1)
    env.updateEnv()
    out = env.step(1, env.agents[0].next_target_ind)
    assert len(out) == 7  # 7-tuple including reward
