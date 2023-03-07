import os
import shutil
import argparse
import configparser
import glob
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import time
from collections import deque
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from algorithms.ppo_patrol import PPO
from buffers.rollout_buffer import RolloutBuffer,RolloutBatch, BATCH_FIELDS
from gym_patrol.envs.MAPatrol import MAPatrolEnv
from models.model_patrol import ActorCriticPolicy

def make_batch(bs, trajectory_list): # 只需要observation, actions, old_log_probs, advantages和returns，因为训练只需要这些，但是这些也是由reward, done等transition field的算出来的
    batch_list = []
    observation_list = []
    for t in trajectory_list:   # trajectory_list有numAgent个元素，每个装了一个agent一整个episode收集的数据，每个数据都是max_time_steps个.
        observation_list.append(t.observations)
    ag_state = np.concatenate([ag for ag, _, _, _ in observation_list], axis=0) # size: (numAgents*max_time_steps, numAgents, 4)
    tg_state = np.concatenate([tg for _, tg, _, _ in observation_list], axis=0) # size: (numAgents*max_time_steps, numTargets, 3)
    cn_state = np.concatenate([cn for _, _, cn, _ in observation_list], axis=0) # size: (numAgents*max_time_steps, numTargets, 4)
    mask_state = np.concatenate([mk for _, _, _, mk in observation_list], axis=0) # size: (numAgents*max_time_steps, numTargets)
    observations = (ag_state, tg_state, cn_state, mask_state)
    # observations = np.concatenate([np.array(list(t.observations)) for t in trajectory_list], axis=0) # 把所有agent一个trajectory的observation都拼接起来
    actions = np.concatenate([t.actions for t in trajectory_list], axis=0)
    old_log_probs = np.concatenate([t.old_log_probs for t in trajectory_list], axis=0)
    advantages = np.concatenate([t.advantages for t in trajectory_list], axis=0)
    returns = np.concatenate([t.returns for t in trajectory_list], axis=0)
    itr = 0
    while True:
        if itr*bs+bs <= actions.shape[0]:
            observation_batch = []
            for item in observations: # turn 'obs' array to 'obs_tensor' torch tensor
                observation_batch.append(item[itr*bs:itr*bs+bs-1])
            batch_list.append(RolloutBatch(observation_batch, actions[itr*bs:itr*bs+bs-1], old_log_probs[itr*bs:itr*bs+bs-1], advantages[itr*bs:itr*bs+bs-1], returns[itr*bs:itr*bs+bs-1]))
            itr += 1
            if itr*bs+bs == actions.shape[0]:
                break
        else:
            observation_batch = []
            for item in observations: # turn 'obs' array to 'obs_tensor' torch tensor
                observation_batch.append(item[itr*bs:])
            batch_list.append(RolloutBatch(observation_batch, actions[itr*bs:], old_log_probs[itr*bs:], advantages[itr*bs:], returns[itr*bs:]))  
            break
    return batch_list

def arg_parse():
    parser = argparse.ArgumentParser(description="Multi-agent Patrolling")
    parser.add_argument("--config", type=str, default="configs/params.cfg")
    parser.add_argument("--section", type=str, default="single_proc_ppo")
    parser.add_argument("--test", action="store_true")   # 对于true true类型参数，一旦有这个参数，若是store_true则一旦命令行传入--test则为true,若不传该参数则默认为false,
    parser.add_argument("--load", action="store_false")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--greedy", type=bool, default=True)
    parser.add_argument("--suffix", type=str, default="v0")
    return parser.parse_args()


def cfg_parse(args):
    cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    cfg.read(args.config)
    if args.section in cfg.sections():
        return cfg[args.section]
    else:
        raise Exception(f"Section {args.section} does not exist!")

def main(args, cfgs, writer=None):
    env = MAPatrolEnv()
    env.params_from_cfg(cfgs)
    env._initEnv()
    if not args.test:
        global_step = 0
        episode = 0
        episode_reward = 0
        episode_rewards = deque(maxlen=cfgs.getint("window_size"))
        max_episode_reward = -np.inf
        max_running_episode_reward = -np.inf
        best_model = None
        print("Initializing learner ------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ppo_learner = PPO(cfgs, device, ActorCriticPolicy)
        if args.load:
            if glob.glob(cfgs.get("model_path")+"/*.pth"):
                print("loading model from {}".format(cfgs.get("model_path")))
                ppo_learner.load(cfgs.get("model_path"), args.suffix)

        trajectories = [RolloutBuffer(cfgs.getfloat("gae_lambda", 1), cfgs.getfloat("gamma", 0.99)) for _ in range(env.numAgents)] # 一个RolloutBuffer存了一个agent在一个episode的整个轨迹
        batch_size = cfgs.getint("batch_size")
        
        while episode < args.num_episodes:
            env.reset(scattered=True)
            done = False
            episode_reward = 0 # 所有agent一个episode的reward和
            for _ in range(env._max_episode_steps):
                # inference
                # episode_reward = 0
                act_list = []
                obs_list = []
                obs_tensor_list = []
                next_obs_list = [] # 每个time step的obs_list和next_obs_list都要清零，所以每个时间步一个list只存了agents个数据
                value_list = []
                act_log_prob_list = []
                reward_list = []
                
                # # 先render看看action对不对
                # env.render(True, True)
                
                # agent都通讯一遍，此时agent_states size更新为(numAgents, 4)
                for id in range(1, env.numAgents + 1):
                    # if env.agents[id].isOnNode:
                    env.communicate(id)
                env.updateEnv()
                
                # 每个agent得到自己的observation并记录下来，第一次时候agent_states的size是(1, 4)
                for id in range(1, env.numAgents + 1):
                    obs = env.observe(id, relative=True)
                    obs_tensor = []
                    for item in obs: # turn 'obs' array to 'obs_tensor' torch tensor
                        obs_tensor.append(torch.FloatTensor(item).to(device))
                    obs_tensor_list.append(tuple(obs_tensor))
                    obs_list.append(obs) # append array形式方便后面rollout buffer直接convert to batch
                    
                for id in range(1, env.numAgents + 1):
                    with torch.no_grad():
                        act_tensor, act_log_prob_tensor, value_tensor = ppo_learner.act(obs_tensor_list[id-1]) # 用tensor形式传进网络
                        act, act_log_prob, value = act_tensor.cpu().numpy(), act_log_prob_tensor.cpu().numpy(), value_tensor.item()
                        act_list.append(act)
                        act_log_prob_list.append(act_log_prob)
                        value_list.append(value)
                    
                for id in range(1, env.numAgents + 1):
                    act = act_list[id-1].squeeze(0)[0] # [0] is for extracting the value                                             
                    next_output = env.step(id, act)
                    agent_states, target_states, connectivity_feature, local_mask, _, _, reward = next_output
                    next_obs = agent_states, target_states, connectivity_feature, local_mask
                    next_obs_list.append(next_obs)
                    episode_reward += reward
                    reward_list.append(reward)
                    
                done = env.checkFinished()
                
                for idx, buffer in enumerate(trajectories): # idx是agent编号
                    # episode_reward += reward / env.numAgents
                    buffer.add(obs_list[idx], act_list[idx], reward_list[idx], done, value_list[idx], act_log_prob_list[idx])
                    
                if done:
                    break
                global_step += 1

            # compute bootstrap value
            trajectory_list = []
            with torch.no_grad():
                for idx, buffer in enumerate(trajectories):
                # 把每个agent的最后一个value拿出来
                    next_obs_tensor = []
                    for item in next_obs_list[idx]: # 把最后一个时间步的next_obs_list转成tensor
                        next_obs_tensor.append(torch.FloatTensor(item).to(device))
                    _, _, last_v = ppo_learner.act(tuple(next_obs_tensor))
                    buffer.post_process(last_v.item())
                    trajectory_list.append(buffer.get_batch())
                    buffer.reset()
            episode += 1
            episode_rewards.append(episode_reward)
            running_episode_reward = np.mean(episode_rewards)
            # store best results
            if max_episode_reward < episode_reward:
                max_episode_reward = episode_reward
            if max_running_episode_reward < running_episode_reward:
                best_model = deepcopy(ppo_learner.get_weights())
                print("[Episode {}|{}] Found better running episode reward: {} than previous one: {}".format(episode, args.num_episodes, running_episode_reward, max_running_episode_reward))
                max_running_episode_reward = running_episode_reward
            
            value_loss = []
            policy_loss = []
            entropy_loss = []
            clip_fraction = []
            approx_kl_div_list = []
            batch_list = make_batch(batch_size, trajectory_list)
            for batch in batch_list:
                v_l, pg_l, ent_l, clip_frac, approx_kl_div = ppo_learner.update(batch)
                value_loss.append(v_l)
                policy_loss.append(pg_l)
                entropy_loss.append(ent_l)
                clip_fraction.append(clip_frac)
                approx_kl_div_list.append(approx_kl_div)
            
            # v_l, pg_l, ent_l, clip_frac, approx_kl_div = ppo_learner.update(buffer)
            buffer.reset() # reset rollout buffer
            writer.add_scalar("Perf/episode_reward", episode_reward, global_step=episode)
            writer.add_scalar("Perf/running_episode_reward", running_episode_reward, global_step=episode)
            writer.add_scalar("Train/value_loss", v_l, global_step=episode)
            writer.add_scalar("Train/policy_loss", pg_l, global_step=episode)
            writer.add_scalar("Train/entropy_loss", ent_l, global_step=episode)
            writer.add_scalar("Train/clip_fraction", clip_frac, global_step=episode)
            writer.add_scalar("Train/approx_kl_div", approx_kl_div, global_step=episode)
            writer.flush()
            print("[Episode {}] value loss: {:.4f}, policy loss: {:.4f}, episode reward: {}, running episode reward: {:.4f}".format(episode, v_l, pg_l, episode_reward, running_episode_reward))
            if episode % 10 == 0:
                print("Progress [Episode {}|{}] --- {:.2f}%".format(episode, args.num_episodes, episode/args.num_episodes*100))
        print("Training finished, save best model ...")
        ppo_learner.set_weights(best_model)
        ppo_learner.save(cfgs.get("model_path"), args.suffix)
    else:
        test_device = torch.device("cpu")
        print("Initializing tester ------")
        tester = PPO(cfgs, test_device, ActorCriticPolicy)
        if glob.glob(cfgs.get("model_path")+"/*.pth"):
            print("loading model from {}".format(cfgs.get("model_path")))
            tester.load(cfgs.get("model_path"), args.suffix)
        test_path = cfgs.get("test_path")
        for i in range(args.num_tests):
            env.reset(scattered=True)
            done = False
            # episode_reward = 0
            # eps_reward = 0
            score = 0
            while True:
                # inference
                act_list = []
                obs_list = []
                if args.render:
                    env.render(True, True)
                if done:
                    env.close()
                    break
                # agent都通讯一遍，此时agent_states size更新为(numAgents, 4)
                for id in range(1, env.numAgents + 1):
                    env.communicate(id)
                env.updateEnv()
                # 每个agent得到自己的observation并记录下来，第一次时候agent_states的size是(1, 4)
                for id in range(1, env.numAgents + 1):
                    obs = env.observe(id, relative=True)
                    obs_tensor = []
                    for item in obs: # turn 'obs' array to 'obs_tensor' torch tensor
                        obs_tensor.append(torch.FloatTensor(item).to(device))
                    obs_tensor_list.append(tuple(obs_tensor))
                    obs_list.append(obs) # append array形式方便后面rollout buffer直接convert to batch
                for id in range(1, env.numAgents + 1):
                    with torch.no_grad():
                        act_tensor, _, _ = tester.act(obs_tensor_list[id-1]) # 用tensor形式传进网络
                        act= act_tensor.cpu().numpy()
                        act_list.append(act)
                        # act_log_prob_list.append(act_log_prob)
                        # value_list.append(value)
                
                
                for id in range(1, env.numAgents + 1):
                    act = act_list[id-1].squeeze(0)[0] # [0] is for extracting the value                                             
                    next_output = env.step(id, act)
                    _, _, _, _, _, _, reward = next_output
                    # next_obs = agent_states, target_states, connectivity_feature
                    # next_obs_list.append(next_obs)
                    score += reward
                env._elapsed_steps += 1
                done = env.checkFinished()
                score = score / env.numAgents
     
            print(f"[Epoch-{i}] Test perf: {score}")
            env.save(f"{test_path}/agents{env.numAgents}-targets{env.numTargets}-comms_radius{env.comms_radius if env.comms_radius is not None else 'inf'}-max_steps{env.max_episode_steps}.gif")



if __name__ == "__main__":
    args = arg_parse()
    cfgs = cfg_parse(args)

    if not args.test:
        if os.path.exists(cfgs.get('train_path')): # 删除的是train_path里的文件非models文件夹
            print(f"Previous training logs removed from {cfgs.get('train_path')}!")
            for root, dirs, files in os.walk(cfgs.get('train_path')):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for dir in dirs:
                    shutil.rmtree(os.path.join(root, dir))
        writer = SummaryWriter(cfgs.get('train_path'))

        if not os.path.exists(cfgs.get('model_path')):
            print("Model path created!")
            os.makedirs(cfgs.get('model_path'))

    if not args.test:
        main(args, cfgs, writer)
    else:
        main(args, cfgs)