import numpy as np
import torch
import os
from buffers.rollout_buffer import RolloutBatch, BATCH_FIELDS
from models.model_patrol import ActorCriticPolicy


def convert_batch_to_tensor(batch, device):
    assert isinstance(batch, RolloutBatch)
    # tensor_dict = {k: torch.FloatTensor(getattr(batch, k)).to(device) for k in BATCH_FIELDS} 
    tensor_dict = {}
    for k in BATCH_FIELDS:
        if k == 'observations':
            observations_tensor = []
            ag, tg, cn, mk = tuple(getattr(batch, k))
            observations_tensor = [torch.FloatTensor(ag).to(device), torch.FloatTensor(tg).to(device), torch.FloatTensor(cn).to(device), torch.FloatTensor(mk).to(device)]
            tensor_dict[k] = observations_tensor
        else:   
            tensor_dict[k] = torch.FloatTensor(getattr(batch, k)).to(device)
    return RolloutBatch(**tensor_dict)


class PPO:
    def __init__(self, cfgs, device, policy: ActorCriticPolicy, phase = 0):
        self.device = device
        self.phase = phase
        self.policy = policy(cfgs).to(device)
        optimizer = getattr(torch.optim, cfgs.get("optim", "Adam")) # getattr: return the given attribute of a class object
        self.lr = cfgs.getfloat("learning_rate")
        # self.optimizer = optimizer(self.policy.parameters(), lr=self.lr*(10**(-phase)))
        self.optimizer = optimizer(self.policy.parameters(), lr=self.lr)
        self.mse_loss = torch.nn.MSELoss()
        self.grad_max = cfgs.getfloat("max_grad")
        self.n_epochs = cfgs.getint("n_epochs")
        self.entropy_coef = cfgs.getfloat("entropy_coef")
        self.value_coef = cfgs.getfloat("value_coef")
        self.surr_clip = cfgs.getfloat("surrogate_loss_clip")

    def get_weights(self):
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'phase': self.phase
            }
        # return self.policy.state_dict()
        return checkpoint

    def set_weights(self, temp_best, load_all = False):     
        if load_all:
            # for _, values in checkpoint.items():
            #     for k, v in values.items():
            #         values[k] = v.to(self.device)
            # for k, v in checkpoint['model_state_dict'].items():
            #     checkpoint['model_state_dict'][k] = v.to(self.device)
            # for k, v in checkpoint['optimizer_state_dict'].items():
            #     checkpoint['optimizer_state_dict'][k] = v.to(self.device)
            # checkpoint['phase'] = checkpoint['phase'].to(self.device)
            self.policy.load_state_dict(temp_best['model_state_dict'])
            self.optimizer.load_state_dict(temp_best['optimizer_state_dict'])
            self.phase = temp_best['phase']
        else:
            weights = temp_best['model_state_dict']
            # for k, v in weights.items():
            #         weights[k] = v.to(self.device)
            self.policy.load_state_dict(weights)

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
        # convert batch to tensor后： observation; action: (255, 1); advantage: (255,); 
        # observations: [agent states (255, numAgents, 4), target states (255, numTarget, 3), connectivity feature (255, numTargets, 4)]; 
        # old_log_prob: (255, 1); returns: (255, )

        for i in range(self.n_epochs):
            # evaluate batch
            values, act_log_prob, act_entropy = self.evaluate(batch.observations, batch.actions)

            advantages = batch.advantages # (B,)
            # normalize advantage
            # advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)

            ratio = torch.exp(act_log_prob-batch.old_log_probs) # i.e. exp(log [p(theta)/p(theta')] ) = p(theta)/p(theta')
                                                                # size of act_log_prob: max_time_steps, i.e. [999]
            surr_loss_1 = ratio * advantages # shape: 999
            surr_loss_2 = torch.clamp(ratio, 1-self.surr_clip, 1+self.surr_clip) * advantages # shape: 999

            policy_loss = -torch.min(surr_loss_1, surr_loss_2).mean() # policy_loss is a scalar

            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.surr_clip).float()).item() # 记录多少个参与了clip
            clip_fractions.append(clip_fraction)

            # value_loss = self.mse_loss(values.squeeze(1), batch.returns)
            value_loss = self.mse_loss(values.squeeze(), batch.returns)
            value_losses.append(value_loss.item())

            entropy_loss = -torch.mean(act_entropy)
            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Calculate approximate form of reverse KL Divergence for early stopping
            # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
            # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
            # and Schulman blog: http://joschu.net/blog/kl-approx.html

            with torch.no_grad(): # KL_div: sum from 1 to N [ p(x) * log (p(x)/q(x))]
                log_ratio = act_log_prob - batch.old_log_probs
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item() # Schulman blog: http://joschu.net/blog/kl-approx.html
                approx_kl_divs.append(approx_kl_div)

            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_max)
            self.optimizer.step()

        self.policy.eval()    

        return np.mean(value_losses), np.mean(pg_losses), np.mean(entropy_losses), np.mean(clip_fractions), np.mean(approx_kl_divs)

    def save(self, model_path, suffix=""):
        model_path = os.path.join(model_path, f"checkpoint_{suffix}.pth")
        model_path = model_path.replace('\\','/')
        print(f"Saving checkpoint to {model_path}")
        # torch.save(self.policy.state_dict(), model_path)
        torch.save({
            'phase': self.phase,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, model_path)

    def load(self, model_path, suffix="", load_all=False):
        model_path = os.path.join(model_path, f"checkpoint_{suffix}.pth")
        model_path = model_path.replace('\\','/')
        print(f"loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path)
        if load_all:
            # self.set_weights(torch)
            self.policy.load_state_dict(checkpoint['model_state_dict']) #.to(self.device) # !!!!
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #.to(self.device)
            self.phase.load_state_dict(checkpoint['phase']) #.to(self.device)
        else:
            # self.set_weights(torch.load(model_path))
            self.policy.load_state_dict(checkpoint['model_state_dict']) #.to(self.device) # !!!!
        # self.set_weights(torch.load(model_path), load_all)
        
   




