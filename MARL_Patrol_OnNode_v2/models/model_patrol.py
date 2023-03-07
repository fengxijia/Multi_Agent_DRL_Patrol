### Part of the code credit to https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.distributions import Categorical

import argparse
import configparser

import sys, os
# sys.path.append("..") 
sys.path.append(os.getcwd())
# # import single_proc_ppo_patrol
# from single_proc_ppo_patrol import arg_parse, cfg_parse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Embedder(nn.Module):
    
    '''
    Correspond to connectivity/target/agent feature embedding.
    A fully connected linear layer.
    
    '''
    
    def __init__(self, d_featureLast, d_model):
        super().__init__()
        self.embed = nn.Linear(d_featureLast, d_model)
    def forward(self, x):
        return self.embed(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads # Division, rounds down
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        # input shape: batchsize * n_sample * d_model
        bs = q.size(0)
        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # 4, 1, 8, 2 
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * heads * sl * d_model
        q = q.transpose(1,2)
        k = k.transpose(1,2) # 4, 8, 1, 2
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        output_attention, _ = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = output_attention.transpose(1,2).contiguous().view(bs, -1, self.d_model) # 4, 1, 16
        output = self.out(concat)
        return output


def attention(q, k, v, d_k, mask, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        # mask = mask.unsqueeze(1)
        if mask.shape != scores.shape:
            mask = mask.view(scores.size(0), 1, scores.size(-2), scores.size(-1)).expand_as(scores)
        scores = scores.masked_fill(mask == 0, -1e9) # Replace all the 0 in 'mask' with -1e9
        scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores # add one more return value

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module): 
    '''
    A basic encoder block which includes two residual blocks.
    The entire encoder consists of N such small encoders.
    
    '''
    def __init__(self, d_model, heads, dropout = 0.1):  # d_model corresponds to the 'd'
        super().__init__()
        # print('EncoderLayer')
        self.norm_1 = Norm(d_model)                     # Corresponds to the first residual block in the encoder
        self.norm_2 = Norm(d_model)                     # Corresponds to the second residual block in the encoder
        ############
        self.norm_3 = Norm(d_model)                     # Corresponds to the second residual block in the encoder
        self.norm_4 = Norm(d_model)                     # Corresponds to the second residual block in the encoder
        ############
        self.attn = MultiHeadAttention(heads, d_model)  # heads, d_model are construction parameters for the MultiHeadAttention layer
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, q, k, v):  # self, q, k, v           # x:(10, 10, 16) ae_output: (4, 4, 16) 要normalize y吗，decoderlayer没有normalize
        q1 = self.norm_1(q) # 10, 16 
        k1 = self.norm_2(k) # 10, 16
        v1 = self.norm_3(v) # 10, 16
        x = q + self.dropout_1(self.attn(q1, k1, v1))     # deleted the mask, q: 10, 16
        x2 = self.norm_4(x)
        x = x + self.dropout_2(self.ff(x2))
        return x # 10, 10, 16
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, e_output, trg_mask): # deleted the src_mask
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, e_output, e_output, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, x2, x2)) # ？？？？？需要加mask吗
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
           
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_featureLast, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(d_featureLast, d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
        
    def forward(self, src): # deleted the mask
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x, x, x) # deleted the mask
        return self.norm(x) # 4,4,16
    
# embed1 = Embedder(*args)
# embed2 = Embedder(*args)

# encoder_targert = Encoder_Target(*d_cf, embed1, embed2)

# class Encoder_Target(nn.Module):
#     def __init__(self, d_cf, d_tf, d_model, N, heads, *embeds, **kwargs):
    
class Encoder_Target(nn.Module):
    def __init__(self, d_cf, d_tf, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed1 = Embedder(d_cf, d_model)
        self.embed2 = Embedder(d_tf, d_model)
        # self.embeds = embeds == []
        # self.embeds[0]
        # self.embeds[1]
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
        
    def forward(self, connect_f, target_f): # deleted the mask
        
        '''
        Add the connectivity and target embeddings.
        
        '''
        ce = self.embed1(connect_f)
        te = self.embed2(target_f)
        x = ce + te
        for i in range(self.N):
            x = self.layers[i](x,x,x)      # deleted the mask
        return self.norm(x)
    
class Encoder_TargetAgent(nn.Module): # combine with the encoder
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        # self.embed = Embedder(d_featureLast, d_model) # 这里没有embedder了如何合并
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
        
    def forward(self, x, ae_output):           # deleted the mask
        for i in range(self.N):
            x = self.layers[i](x, ae_output, ae_output)   # deleted the mask. x:(10, 10, 16) ae_output: (4, 4, 16)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        # self.embed = Embedder(d_featureLast, d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, x, e_output, trg_mask):
        for i in range(self.N):
            x = self.layers[i](x, e_output, trg_mask)
        return self.norm(x)

# ### Haven't figured out how to find the current agent embedding from all agent embeddings, so just take the first value as the current embedding for now.
def index_pooling(ae_output, te_output):
    indexing = ae_output[:,:1,:]    # Haven't figured out how to find the current agent embedding from all agent embeddings,
                                    # So just take the first value as the current embedding for now.
                                    # Use :1 instead of 0. Indexing will reduce the dimensions.
    mean_pooling = torch.mean(te_output, 1, True)
    output = mean_pooling + indexing
    return output

class ActorCriticPolicy(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.d_model = cfgs.getint("dim_unify")
        self.N = cfgs.getint("repeat_times_inside_net")
        self.heads = cfgs.getint("heads")
        self.dim_af = cfgs.getint("dim_af")
        self.dim_cf = cfgs.getint("dim_cf")
        self.dim_tf = cfgs.getint("dim_tf")
        self.agent_encoder = Encoder(self.dim_af, self.d_model, self.N, self.heads)
        self.target_encoder = Encoder_Target(self.dim_cf, self.dim_tf, self.d_model, self.N, self.heads)
        self.tgt_agt_encoder = Encoder_TargetAgent(self.d_model, self.N, self.heads)
        self.decoder = Decoder(self.d_model, self.N, self.heads)
        # self.pointer_net = Decoder(d_model, N, heads)
        self.pointer_net = attention
        self.linear_critic = nn.Linear(self.d_model, 1)
        # self.softmax = nn.Softmax(dim=2)
        # self.linear_policy = nn.Linear(d_model, n_target)
         
    # def forward(self, observation, trg_mask=None):
    def forward(self, observation):
        agent_f, target_f, connect_f, trg_mask = observation
        ae_output = self.agent_encoder(agent_f)
        te_output = self.target_encoder(connect_f, target_f)
        tae_output = self.tgt_agt_encoder(te_output, ae_output)
        x = index_pooling(ae_output, te_output) #ae: 4, 4, 16
        output_decoder = self.decoder(x, tae_output, trg_mask)
        critic_value = self.linear_critic(output_decoder)
        _, policy = self.pointer_net(output_decoder, tae_output, tae_output, self.d_model, trg_mask)
        # policy = self.softmax(policy)
        # policy = policy.squeeze(0)
        # output = self.linear_policy(output_pointer)
        return critic_value, policy

    def evaluate(self, obs, act):
        value, policy = self.forward(obs)
        act_dist = Categorical(policy)
        act_log_prob = torch.sum(act_dist.log_prob(act), dim=-1)
        act_entropy = torch.sum(act_dist.entropy(), dim=-1)
        return value, act_log_prob, act_entropy

    def act(self, obs, greedy=False):
        value, policy = self.forward(obs)
        act_dist = Categorical(policy)
        if greedy:
            # act = act_dist.mean
            act = torch.argmax(policy)
        else:
            act = act_dist.sample() # act: torch.size([1,1])
        act_log_prob = torch.sum(act_dist.log_prob(act), dim=-1) # act_log_prob: torch.size([1])
        return act, act_log_prob, value
    
# ############################################
# def arg_parse():
#     parser = argparse.ArgumentParser(description="Multi-agent Patrolling")
#     parser.add_argument("--config", type=str, default="configs/params.cfg")
#     parser.add_argument("--section", type=str, default="single_proc_ppo")
#     parser.add_argument("--test", action="store_true") #对于true false类型参数，一旦有这个参数，若是store_true就设成true，默认为false
#     parser.add_argument("--load", action="store_true")
#     # parser.add_argument("--total_steps", type=int, default=int(3e6))
#     parser.add_argument("--num_episodes", type=int, default=3000)
#     parser.add_argument("--num_tests", type=int, default=5)
#     parser.add_argument("--render", action="store_true")
#     parser.add_argument("--greedy", type=bool, default=True)
#     parser.add_argument("--suffix", type=str, default="v0")
#     return parser.parse_args()


# def cfg_parse(args):
#     cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
#     cfg.read(args.config)
#     if args.section in cfg.sections():
#         return cfg[args.section]
#     else:
#         raise Exception(f"Section {args.section} does not exist!")
# ############################################

if __name__ == "__main__":
    from single_proc_ppo_patrol import arg_parse, cfg_parse
    args = arg_parse()
    cfgs = cfg_parse(args)
    net = ActorCriticPolicy(cfgs)
    num_agents = 4
    num_targets = 10
    batch_size = 40
    obs = (torch.randn(batch_size,num_agents, 4), torch.randn(batch_size,num_targets, 3), torch.randn(batch_size,num_targets, 4), torch.tensor(batch_size*[[[True]*num_targets]])) # agent, target, connect, mask(batch size, shape (1, numTargets))
    critic_value, policy = net(obs)
    # print(policy.squeeze())
    # print(policy.sum())
    print('Value', critic_value.shape)
    print('Policy', policy.shape)
    