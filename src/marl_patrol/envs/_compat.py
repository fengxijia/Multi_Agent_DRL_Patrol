"""Gym/Gymnasium compatibility shim.

The environments only need ``gym.Env`` as a base class (their reset/step are
task-specific and never go through the standard gym API), so either the modern
``gymnasium`` package or the legacy ``gym`` package works.
"""

try:  # modern stack
    import gymnasium as gym
except ImportError:  # pragma: no cover - legacy stack
    import gym

__all__ = ["gym"]
