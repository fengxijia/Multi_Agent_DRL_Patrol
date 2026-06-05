"""On-policy rollout buffer and the batch containers used by PPO.

One :class:`RolloutBuffer` stores the full trajectory of a single agent for one
episode. After the episode it computes GAE advantages and discounted returns in
:meth:`RolloutBuffer.post_process`, then exposes the data as a
:class:`RolloutBatch` of stacked numpy arrays for the learner.
"""

from collections import namedtuple

import numpy as np

# Per-timestep fields stored while collecting a trajectory.
TRANSITION_FIELDS = (
    "obs",
    "action",
    "reward",
    "done",
    "value",
    "log_prob",
    "discount_return",
    "gae",
)
# Fields actually consumed by the PPO update.
BATCH_FIELDS = ("observations", "actions", "old_log_probs", "advantages", "returns")

# namedtuple behaves like a dict but also supports unpacking/iteration.
RolloutBatch = namedtuple("RolloutBatch", BATCH_FIELDS)


def convert_to_batch(transitions):
    """Stack a list of :class:`OnPolicyTransition` into a :class:`RolloutBatch`."""
    ag_list, tg_list, cn_list, mask_list = [], [], [], []
    for t in transitions:
        ag_s, tg_s, cn_s, mask = t.obs
        ag_list.append(ag_s)
        tg_list.append(tg_s)
        cn_list.append(cn_s)
        mask_list.append(mask)
    ag_state = np.concatenate(ag_list, axis=0)
    tg_state = np.concatenate(tg_list, axis=0)
    cn_state = np.concatenate(cn_list, axis=0)
    masks = np.concatenate(mask_list, axis=0)
    observations = (ag_state, tg_state, cn_state, masks)

    actions = np.concatenate([t.action for t in transitions], axis=0)
    old_log_probs = np.array([t.log_prob for t in transitions])
    advantages = np.array([t.gae for t in transitions])
    returns = np.array([t.discount_return for t in transitions])
    return RolloutBatch(observations, actions, old_log_probs, advantages, returns)


class OnPolicyTransition:
    """A mutable container for a single transition's fields."""

    def __init__(self, **kwargs):
        super().__init__()
        for k in TRANSITION_FIELDS:
            self.__setattr__(k, None)
        self.init(**kwargs)

    def init(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


class RolloutBuffer:
    """Collects one agent's episode and turns it into training batches.

    Args:
        gae_lambda: GAE lambda. 0 -> one-step TD (low variance, high bias),
            1 -> Monte-Carlo (high variance, low bias).
        gamma: discount factor.
    """

    def __init__(self, gae_lambda, gamma):
        super().__init__()
        self._episode = []
        self.gae_lambda = gae_lambda
        self.gamma = gamma

    def add(self, obs, action, reward, done, value, log_prob):
        self._episode.append(
            OnPolicyTransition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )
        )

    def pop(self, num_step):
        """Drop the last ``num_step`` transitions (used to align rewards)."""
        del self._episode[-num_step:]

    def size(self):
        return len(self._episode)

    def get(self):
        return self._episode

    def reset(self):
        self._episode.clear()

    def post_process(self, last_v):
        """Compute GAE advantages and discounted returns in-place."""
        gae = 0
        discount_return = last_v
        for step in reversed(range(self.size())):
            done = float(self._episode[step].done)
            discount_return = self._episode[step].reward + self.gamma * discount_return * (1 - done)
            td_v = (
                self._episode[step].reward
                + self.gamma * last_v * (1 - done)
                - self._episode[step].value
            )
            last_v = self._episode[step].value
            gae = td_v + self.gae_lambda * self.gamma * gae * (1 - done)
            self._episode[step].gae = gae
            self._episode[step].discount_return = discount_return

    def get_batch(self):
        return convert_to_batch(self._episode)
