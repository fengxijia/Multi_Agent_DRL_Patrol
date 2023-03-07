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
from collections import defaultdict

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
    parser = argparse.ArgumentParser(description = "Multi-agent Patrolling")
    parser.add_argument("--config", type = str, default = "configs/params.cfg")
    parser.add_argument("--section", type = str, default = "single_proc_ppo")
    parser.add_argument("--test", action = "store_true") # Take store_true as exp, as long as the argument presents，the value is true, if not existes, then it's false
    parser.add_argument("--load", action = "store_true")
    parser.add_argument("--fixedMap", action = "store_true")
    parser.add_argument("--num_agents", type = int, default = 4)
    parser.add_argument("--num_targets", type = int, default = 10)
    parser.add_argument("--num_episodes", type = int, default = 10000)
    parser.add_argument("--episode_segment", type = int, default = 200)
    parser.add_argument("--num_tests", type = int, default = 5)
    parser.add_argument("--render", action = "store_true")
    parser.add_argument("--greedy", type = bool, default = True)
    parser.add_argument("--suffix", type = str, default = "v2")
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
    env.params_from_cfg(args, cfgs)
    env._initEnv()
    if not args.test:
        global_step = 0
        episode = 0
        phase = 0
        episode_reward = 0
        episode_rewards = deque(maxlen=cfgs.getint("window_size"))
        max_episode_reward = -np.inf
        max_running_episode_reward = -np.inf
        best_model = None
        print("Initializing learner ------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ppo_learner = PPO(cfgs, device, ActorCriticPolicy)
        # lr_dynamic = ppo_learner.lr*(10**phase)
        # ppo_learner.lr = lr_dynamic
        if args.load:
            if glob.glob(cfgs.get("model_path")+"/*.pth"):
                print("loading model from {}".format(cfgs.get("model_path")))
                ppo_learner.load(cfgs.get("model_path"), suffix=args.suffix, load_all = True)
        trajectories = [RolloutBuffer(cfgs.getfloat("gae_lambda", 1), cfgs.getfloat("gamma", 0.99)) for _ in range(env.numAgents)] # 一个RolloutBuffer存了一个agent在一个episode的整个轨迹
        batch_size = cfgs.getint("batch_size")
        
        while episode < args.num_episodes:
            env.reset(env.fixedMap, scattered=True)
            done = False
            episode_reward = 0 # 所有agent一个episode里的reward和
            episode_onNode = np.zeros(env.numAgents) # 指示一个episode里哪些agent在点上(可能有的agent一个episode都在边上，没有收集数据)
            next_obs_dict  = defaultdict(list)
            for _ in range(env._max_episode_steps):

                # 默认训练时候不render
                if args.render:
                    env.render(True, True)
                
                # 如果都不在点上
                if not env.checkAnyOnNode(): # 如果都不在Node上
                    for id in range(1, env.numAgents + 1):
                        act = env.agents[id-1].next_target_ind
                        env.agents[id-1].move(act)
                    env.tick()
                    continue
                
                else:
                    ## 如果有agent在点上
                    # list再清零
                    act_list = []
                    obs_list = []
                    obs_tensor_list = []
                    # next_obs_list = [] # 每个time step的obs_list和next_obs_list都要清零，所以每个时间步一个list只存了agents个数据
                    value_list = []
                    act_log_prob_list = []
                    reward_list = []
                    onNode_agents = [] # idx started from 1
                    onEdge_agents = [i+1 for i in range(env.numAgents)] # started from 1
                    
                    # 在node的agent都通讯一遍，此时agent_states size更新为(numAgents, 4)
                    for id in range(1, env.numAgents + 1):
                        if env.agents[id-1].isOnNode:
                            env.communicate(id)
                    env.updateEnv()
                    
                    # 每个agent得到自己的observation并记录下来，第一次时候agent_states的size是(1, 4)
                    for id in range(1, env.numAgents + 1):
                        if env.agents[id-1].isOnNode:
                            onNode_agents.append(id)
                            obs = env.observe(id, relative=True)
                            obs_tensor = []
                            for item in obs: # turn 'obs' array to 'obs_tensor' torch tensor
                                obs_tensor.append(torch.FloatTensor(item).to(device))
                            # obs_tensor_list.append(tuple(obs_tensor))
                            obs_tensor_list.append(obs_tensor)
                            obs_list.append(obs) # append array形式方便后面rollout buffer直接convert to batch
                        
                    ag_states = torch.cat([ag for ag, _, _, _ in obs_tensor_list], axis=0)
                    tg_states = torch.cat([tg for _, tg, _, _ in obs_tensor_list], axis=0) # size: (numAgents*max_time_steps, numTargets, 3)
                    cn_states = torch.cat([cn for _, _, cn, _ in obs_tensor_list], axis=0) # size: (numAgents*max_time_steps, numTargets, 4)
                    mask_states = torch.cat([mk for _, _, _, mk in obs_tensor_list], axis=0) # size: (numAgents*max_time_steps, numTargets)
                    obs_batch = (ag_states, tg_states, cn_states, mask_states) # 存储顺序和onNode_agents的顺序一致
                        
                    
                    # for i, id in enumerate(onNode_agents):
                    with torch.no_grad():
                        act_tensor, act_log_prob_tensor, value_tensor = ppo_learner.act(obs_batch) # 用tensor形式传进网络
                        act_array, act_log_prob_array, value_array = act_tensor.cpu().numpy(), act_log_prob_tensor.cpu().numpy(), value_tensor.cpu().numpy() #都是行向量的形状
                        act_list.append(act_array)
                        act_log_prob_list.append(act_log_prob_array)
                        value_list.append(value_array)
                    
                    for i, id in enumerate(onNode_agents):
                        act = act_array[i].item() # [0] is for extracting the value                                             
                        reward = env.get_agent_reward(id, localReward=False)
                        env.agents[id-1].move(act)
                        next_obs = env.observe(id)
                        next_obs_dict[id-1].append(next_obs)
                        episode_reward += reward
                        reward_list.append(reward)
                    
                    # for id in range(1, env.numAgents + 1):
                    #     env.agents[id-1].move()
                    # env.tick()
                    
                    for i in onNode_agents: # 提取哪些agent在该time step一直都在边上
                        onEdge_agents.remove(i)
                    for i, id in enumerate(onEdge_agents):
                        act = env.agents[id-1].next_target_ind
                        env.agents[id-1].move(act)    
                    done = env.checkFinished()
                    
                    for i, id in enumerate(onNode_agents):
                        trajectories[id-1].add(obs_list[i], act_list[0][i], reward_list[i], done, value_list[0][i], act_log_prob_list[0][i])
                        
                    if done:
                        break
                    global_step += 1
                    
                    # step_valid_indice = []
                    # for id in onNode_agents:
                    #     if len(trajectories[id-1].get()) > 1:
                    #         step_valid_indice.append(id-1)
                    # episode_valid[step_valid_indice] = 1 # 每个时间步检查一下哪些agent在点上过且收集了至少一个有效数据，合并到整个episode里在点上出现过的list里
                    
                    step_onNode_indice = [i-1 for i in onNode_agents] # started from 0!
                    episode_onNode[step_onNode_indice] = 1 # 每个时间步检查一下哪些agent在点上过，合并到整个episode里在点上出现过的list里
            
                        
            # 一个episode的数据收集结束了
            # 只有当episode中有agent在点上过，才做数据后处理
            episode_onNode_indice = list(np.array(np.where(episode_onNode==1))[0]) # started from 0!
            # Move reward one step forward, cut out the last data
            episode_valid_indice = [] # 记录一个episode中有有效数据的agent index
            if len(episode_onNode_indice) > 0:
                for i, idx in enumerate(episode_onNode_indice):
                    if len(trajectories[idx].get()) > 1: # 一个episode收集大于一个数据才是有效数据（reward问题）
                        episode_valid_indice.append(idx)
                        last_time_data = trajectories[idx].get()[-1]
                        obs_pse, action_pse, reward_pse, done_pse, value_pse, log_prob_pse = last_time_data.obs, last_time_data.action, last_time_data.reward, last_time_data.done, last_time_data.value, last_time_data.log_prob # copy the last time step data, so as to move reward ahead
                        trajectories[idx].add(obs_pse, action_pse, reward_pse, done_pse, value_pse, log_prob_pse) # add pseudo time step data for each agent
                        data_list = trajectories[idx].get()
                        for j in range(0, len(data_list)-1):
                            data_list[j].reward = data_list[j+1].reward
                        trajectories[idx].pop(2) # delete the last two time steps
                    else:
                        trajectories[idx].pop(1)
                
                trajectory_list = []
                with torch.no_grad():
                    for i, idx in enumerate(episode_valid_indice):
                        # 把每个agent的最后一个value拿出来
                        # if len(trajectories[idx].get()) > 0:
                        buffer = trajectories[idx]
                        next_obs_tensor = []
                        for item in next_obs_dict[idx][-1]: # 把最后一个时间步的next_obs_list转成tensor
                            next_obs_tensor.append(torch.FloatTensor(item).to(device))
                        _, _, last_v = ppo_learner.act(tuple(next_obs_tensor))
                        buffer.post_process(last_v.item())
                        trajectory_list.append(buffer.get_batch())
                        buffer.reset()
                        # else:
                        #     continue
                episode += 1
                episode_rewards.append(episode_reward)
                running_episode_reward = np.mean(episode_rewards)
                # store best results
                if max_episode_reward < episode_reward:
                    max_episode_reward = episode_reward
                if max_running_episode_reward < running_episode_reward:
                    best_model = deepcopy(ppo_learner.get_weights())
                    print("[Episode {}|{}] Found better running episode reward: {:.4f} than previous one: {}".format(episode, args.num_episodes, running_episode_reward, max_running_episode_reward))
                    max_running_episode_reward = running_episode_reward
                
                if len(episode_valid_indice) > 0:
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
                    print("[Episode {}] value loss: {:.4f}, policy loss: {:.4f}, episode reward: {:.4f}, running episode reward: {:.4f}".format(episode, v_l, pg_l, episode_reward, running_episode_reward))
                    if episode % 10 == 0:
                        print("Progress [Episode {}|{}] --- {:.2f}%".format(episode, args.num_episodes, episode/args.num_episodes*100))
                else:
                    continue
            else:
                continue
            if episode % args.episode_segment == 0:
                phase += 1
                print("Phase training finished, save best model ...")
                ppo_learner.set_weights(best_model)
                ppo_learner.save(cfgs.get("model_path"), phase)
                ppo_learner.lr *= 0.1
        print("Training finished, save best model ...")
        ppo_learner.set_weights(best_model, load_all=True)
        ppo_learner.save(cfgs.get("model_path"), args.suffix)
    else:
        test_device = torch.device("cpu")
        print("Initializing tester ------")
        tester = PPO(cfgs, test_device, ActorCriticPolicy)
        if glob.glob(cfgs.get("model_path")+"/*.pth"):
            print("loading model from {}".format(cfgs.get("model_path")))
            tester.load(cfgs.get("model_path"), args.suffix)
        test_path = cfgs.get("test_path")
        for eps in range(args.num_tests):
            env.reset(scattered=True)
            done = False
            score = 0
            # while True:
            for _ in range(env._max_episode_steps):
                # inference
                act_list = []
                obs_list = []
                obs_tensor_list = []
                onNode_agents = []
                onEdge_agents = [i+1 for i in range(env.numAgents)] # started from 1
                
                # if args.render:
                # Default: test mode, render
                env.render(True, True)
                if done:
                    env.close()
                    break
                
                # agent都通讯一遍，此时agent_states size更新为(numAgents, 4)
                for id in range(1, env.numAgents + 1):
                    if env.agents[id-1].isOnNode:
                        onNode_agents.append(id)
                        env.communicate(id)
                env.updateEnv()
                
                for i in onNode_agents: # 提取哪些agent在该time step一直都在边上
                    onEdge_agents.remove(i)
                 
                #  if env.agents[id-1].isOnNode:
                #             onNode_agents.append(id)
                #             obs = env.observe(id, relative=True)
                #             obs_tensor = []
                #             for item in obs: # turn 'obs' array to 'obs_tensor' torch tensor
                #                 obs_tensor.append(torch.FloatTensor(item).to(device))
                #             # obs_tensor_list.append(tuple(obs_tensor))
                #             obs_tensor_list.append(obs_tensor)
                #             obs_list.append(obs) # append array形式方便后面rollout buffer直接convert to batch
                            
                for id in range(1, env.numAgents + 1):
                    if env.agents[id-1].isOnNode:
                # for id in onNode_agents:
                        obs_tensor = []
                        obs = env.observe(id, relative=True)
                        for item in obs: # turn 'obs' array to 'obs_tensor' torch tensor
                            obs_tensor.append(torch.FloatTensor(item).to(test_device))
                        obs_tensor_list.append(tuple(obs_tensor))
                        obs_list.append(obs) # append array形式方便后面rollout buffer直接convert to batch
                    else:
                        continue
                
                # for id in range(1, env.numAgents + 1):
                #     # if env.agents[id-1].isOnNode:
                # # for i, id in enumerate(onNode_agents):
                with torch.no_grad():
                    for i, id in enumerate(onNode_agents):
                        act_tensor, _, _ = tester.act(obs_tensor_list[i]) # 用tensor形式传进网络
                        act = act_tensor.cpu().squeeze().numpy().item()
                        reward = env.get_agent_reward(id, localReward=False)
                        env.agents[id-1].move(act)
                        score += reward
                    for i, id in enumerate(onEdge_agents):
                        act = env.agents[id-1].next_target_ind
                        env.agents[id-1].move(act)

                # for id in range(1, env.numAgents + 1):
                #     act = act_list[id-1].squeeze(0)[0] # [0] is for extracting the value                                             
                #     # next_output = env.move(id, act)
                #     # _, _, _, _, _, _, reward = next_output
                #     env.agents[id-1].move(act)
                #     reward = env.get_agent_reward(id)
                #     # next_obs = agent_states, target_states, connectivity_feature
                #     # next_obs_list.append(next_obs)
                #     score += reward

                done = env.checkFinished()
                score = score / env.numAgents
            print(f"[Test episode-{eps}] Test perf: {score}")
            env.save(f"{test_path}/agents{env.numAgents}-targets{env.numTargets}-comms_radius{env.comms_radius if env.comms_radius is not None else 'inf'}-test{eps}.gif")



if __name__ == "__main__":
    args = arg_parse()
    cfgs = cfg_parse(args)

    if not args.test:
        if os.path.exists(cfgs.get('train_path')):
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
    
    ###########################
    else:
        if not os.path.exists(cfgs.get('test_path')):
            os.makedirs(cfgs.get('test_path'))
    ###########################

    if not args.test:
        main(args, cfgs, writer)
    else:
        main(args, cfgs)