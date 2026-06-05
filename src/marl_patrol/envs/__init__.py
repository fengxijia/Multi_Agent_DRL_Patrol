"""Patrolling environments.

Two decision paradigms share the same world dynamics but differ in *when* agents
act:

- :class:`OnNodePatrolEnv` (``on_node``): agents only choose a next target when
  they are physically on a node; between nodes they coast. Supports a
  ``fixed_map`` option to reuse the same graph across episodes.
- :class:`EveryTimeStepPatrolEnv` (``every_timestep``): every agent picks an
  action at every simulation step via :meth:`step`.
"""

from marl_patrol.envs.every_timestep import MAPatrolEnv as EveryTimeStepPatrolEnv
from marl_patrol.envs.on_node import MAPatrolEnv as OnNodePatrolEnv

__all__ = ["OnNodePatrolEnv", "EveryTimeStepPatrolEnv", "make_env"]

# Optional gymnasium registry entries. The envs are normally constructed
# directly (their reset/step signatures are task-specific, not the standard gym
# API), so registration is best-effort and never fatal.
try:  # pragma: no cover - registration is a convenience only
    from marl_patrol.envs._compat import gym

    gym.register(id="MAPatrol-OnNode-v0", entry_point="marl_patrol.envs.on_node:MAPatrolEnv")
    gym.register(
        id="MAPatrol-EveryTimeStep-v0",
        entry_point="marl_patrol.envs.every_timestep:MAPatrolEnv",
    )
except Exception:  # noqa: BLE001 - registry missing or id already registered
    pass


def make_env(mode):
    """Return the env class for ``mode`` in ``{"on_node", "every_timestep"}``."""
    if mode == "on_node":
        return OnNodePatrolEnv
    if mode == "every_timestep":
        return EveryTimeStepPatrolEnv
    raise ValueError(f"Unknown mode {mode!r}; expected 'on_node' or 'every_timestep'.")
