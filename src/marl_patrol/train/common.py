"""Shared helpers for the training / evaluation entry points.

This module collects logic that used to be copy-pasted into every variant's
``single_proc_ppo_patrol.py``: argument parsing, config parsing, mini-batching,
device selection, seeding, logging and run-directory setup.
"""

import argparse
import configparser
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from marl_patrol.buffers.rollout_buffer import RolloutBatch

# Repository root (……/src/marl_patrol/train/common.py -> parents[3]).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "params.cfg"

LOGGER = logging.getLogger("marl_patrol")


def setup_logging(level=logging.INFO):
    """Configure a simple console logger (idempotent)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    LOGGER.setLevel(level)
    return LOGGER


def arg_parse(argv=None):
    """Parse command-line arguments shared by both training modes."""
    parser = argparse.ArgumentParser(description="Multi-agent Patrolling (PPO)")
    parser.add_argument(
        "--mode",
        choices=["on_node", "every_timestep"],
        default="on_node",
        help="Decision paradigm: act only on nodes, or act every timestep.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the .cfg file (defaults to the packaged configs/params.cfg).",
    )
    parser.add_argument("--section", type=str, default="single_proc_ppo")
    parser.add_argument("--test", action="store_true", help="Run evaluation instead of training.")
    parser.add_argument("--load", action="store_true", help="Resume from a saved checkpoint.")
    parser.add_argument(
        "--fixed-map",
        dest="fixed_map",
        action="store_true",
        help="(on_node only) reuse the same graph across episodes.",
    )
    parser.add_argument("--num-episodes", dest="num_episodes", type=int, default=10000)
    parser.add_argument(
        "--episode-segment",
        dest="episode_segment",
        type=int,
        default=200,
        help="(on_node only) episodes between phase checkpoints / LR decay.",
    )
    parser.add_argument("--num-tests", dest="num_tests", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Render during training/eval.")
    # Fix for the argparse ``type=bool`` footgun: use a proper boolean flag.
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Greedy action selection at test time (--no-greedy to sample).",
    )
    parser.add_argument("--suffix", type=str, default="v2", help="Checkpoint suffix to save/load.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    return parser.parse_args(argv)


def cfg_parse(args):
    """Read the requested section of the config file into a config proxy."""
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Pass --config or run from the repo root."
        )
    cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    cfg.read(config_path)
    if args.section not in cfg.sections():
        raise KeyError(f"Section {args.section!r} does not exist in {config_path}.")
    return cfg[args.section]


def get_device(force_cpu=False):
    """Return the torch device, preferring CUDA unless ``force_cpu``."""
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Seed numpy and torch RNGs for reproducibility (no-op if seed is None)."""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(bs, trajectory_list):
    """Concatenate per-agent trajectories and split into mini-batches of size ``bs``.

    The observation tuple ``(agent, target, connectivity, mask)`` is stacked
    field-by-field. Compared with the original implementation this fixes an
    off-by-one slice (``itr*bs+bs-1``) that silently dropped one sample per full
    batch and could emit an empty trailing batch; here every sample is covered
    exactly once and the last (possibly smaller) batch is kept.
    """
    observation_list = [t.observations for t in trajectory_list]
    ag_state = np.concatenate([ag for ag, _, _, _ in observation_list], axis=0)
    tg_state = np.concatenate([tg for _, tg, _, _ in observation_list], axis=0)
    cn_state = np.concatenate([cn for _, _, cn, _ in observation_list], axis=0)
    mask_state = np.concatenate([mk for _, _, _, mk in observation_list], axis=0)
    observations = (ag_state, tg_state, cn_state, mask_state)

    actions = np.concatenate([t.actions for t in trajectory_list], axis=0)
    old_log_probs = np.concatenate([t.old_log_probs for t in trajectory_list], axis=0)
    advantages = np.concatenate([t.advantages for t in trajectory_list], axis=0)
    returns = np.concatenate([t.returns for t in trajectory_list], axis=0)

    n = actions.shape[0]
    batch_list = []
    for start in range(0, n, bs):
        end = min(start + bs, n)
        obs_batch = [item[start:end] for item in observations]
        batch_list.append(
            RolloutBatch(
                obs_batch,
                actions[start:end],
                old_log_probs[start:end],
                advantages[start:end],
                returns[start:end],
            )
        )
    return batch_list


def prepare_run(args, cfgs):
    """Create output dirs and return a TensorBoard writer (None in test mode).

    Mirrors the directory bookkeeping previously duplicated in each variant's
    ``__main__`` block: clear stale training logs, (re)create the model dir, or
    create the test dir for evaluation.
    """
    if not args.test:
        train_path = cfgs.get("train_path")
        if os.path.exists(train_path):
            LOGGER.info("Clearing previous training logs in %s", train_path)
            for root, dirs, files in os.walk(train_path):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(train_path)
        model_path = cfgs.get("model_path")
        if not os.path.exists(model_path):
            LOGGER.info("Creating model dir %s", model_path)
            os.makedirs(model_path)
        return writer

    test_path = cfgs.get("test_path")
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    return None
