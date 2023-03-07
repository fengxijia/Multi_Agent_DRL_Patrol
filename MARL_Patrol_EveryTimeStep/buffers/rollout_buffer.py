from collections import namedtuple
import numpy as np


TRANSITION_FIELDS = ('obs', 'action', 'reward', 'done', 'value', 'log_prob', 'discount_return', 'gae')
BATCH_FIELDS = ('observations', 'actions', 'old_log_probs', 'advantages', 'returns')
RolloutBatch = namedtuple('RolloutBatch', BATCH_FIELDS) # Name: RolloutBatch, keys: BATCH_FIELDS. 
                                                        # Like dictionary, but supports both access from key-value and iteration (dictionaries lack)


def convert_to_batch(transitions):
    # observations = []
    ag_list = []
    tg_list = []
    cn_list = []
    mask_list = []
    for t in transitions:
        ag_s, tg_s, cn_s, mask = t.obs
        ag_list.append(ag_s)
        tg_list.append(tg_s)
        cn_list.append(cn_s)
        mask_list.append(mask)
    ag_state = np.concatenate([ag for ag in ag_list], axis=0)
    tg_state = np.concatenate([tg for tg in tg_list], axis=0)
    cn_state = np.concatenate([cn for cn in cn_list], axis=0)
    masks = np.concatenate([mk for mk in mask_list], axis=0)
    observations = (ag_state, tg_state, cn_state, masks)
    # observations = np.concatenate([t.obs for t in transitions], axis=0) # axis = 0: vertical concat
    actions = np.concatenate([t.action for t in transitions], axis=0)
    old_log_probs = np.array([t.log_prob for t in transitions])
    advantages = np.array([t.gae for t in transitions])
    returns = np.array([t.discount_return for t in transitions])
    return RolloutBatch(observations, actions, old_log_probs, advantages, returns) # Assign values to the initialized namedtuple


class OnPolicyTransition(object):
    def __init__(self, **kwargs):
        super().__init__()
        for k in TRANSITION_FIELDS:
            self.__setattr__(k, None)
        self.init(**kwargs)

    def init(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


class RolloutBuffer(object):
    def __init__(self, gae_lambda, gamma): # gl:0 one-step td error lv hb 1: MC hv lb
        super().__init__()
        self._episode = []
        self.gae_lambda = gae_lambda
        self.gamma = gamma

    def add(self, obs, action, reward, done, value, log_prob):
        self._episode.append(OnPolicyTransition(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob)) # 括号里的参数其实是字典，等号左边是key(k)右边是value(v)

    def size(self):
        return len(self._episode)

    def get(self):
        return self._episode

    def reset(self):
        self._episode.clear()

    def post_process(self, last_v):
        gae = 0
        discount_return = last_v
        for step in reversed(range(self.size())):
            discount_return = self._episode[step].reward + self.gamma * discount_return * (1 - float(self._episode[step].done))
            td_v = self._episode[step].reward + self.gamma * last_v * (1 - float(self._episode[step].done)) - self._episode[step].value
            last_v = self._episode[step].value
            gae = td_v + self.gae_lambda * self.gamma * gae * (1 - float(self._episode[step].done))
            self._episode[step].gae = gae
            # self._episode[step].discount_return = gae + self._episode[step].value
            self._episode[step].discount_return = discount_return

    def get_batch(self):
        return convert_to_batch(self._episode)
