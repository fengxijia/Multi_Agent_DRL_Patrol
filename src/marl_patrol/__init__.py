"""Multi-Agent Deep Reinforcement Learning for the patrolling problem (MRPP).

A single installable package that unifies what used to be three duplicated
project folders (``OnNode_v2``, ``OnNode_FixedMap`` and ``EveryTimeStep``).

Sub-packages
------------
- :mod:`marl_patrol.models`      shared Transformer actor-critic network.
- :mod:`marl_patrol.algorithms`  shared PPO learner.
- :mod:`marl_patrol.buffers`     shared on-policy rollout buffer.
- :mod:`marl_patrol.envs`        the two patrolling environments
  (:class:`~marl_patrol.envs.on_node.MAPatrolEnv` and
  :class:`~marl_patrol.envs.every_timestep.MAPatrolEnv`).
- :mod:`marl_patrol.train`       training / evaluation loops for both modes.
"""

__version__ = "0.2.0"
