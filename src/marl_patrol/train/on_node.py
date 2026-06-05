"""Training / evaluation loop for the *on-node* decision paradigm.

Agents only collect experience when they are on a target node; off-node agents
keep coasting toward their current target. The PPO update and reward logic are
unchanged from the original ``MARL_Patrol_OnNode_v2`` script -- only imports,
device handling, batching and logging were refactored.
"""

import glob
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
import torch

from marl_patrol.algorithms.ppo import PPO
from marl_patrol.buffers.rollout_buffer import RolloutBuffer
from marl_patrol.envs.on_node import MAPatrolEnv
from marl_patrol.models.patrol_net import ActorCriticPolicy
from marl_patrol.train.common import LOGGER, get_device, make_batch


def main(args, cfgs, writer=None):
    env = MAPatrolEnv()
    env.params_from_cfg(cfgs)
    env._initEnv()

    if not args.test:
        _train(args, cfgs, env, writer)
    else:
        _evaluate(args, cfgs, env)


def _train(args, cfgs, env, writer):
    global_step = 0
    episode = 0
    phase = 0
    episode_rewards = deque(maxlen=cfgs.getint("window_size"))
    max_episode_reward = -np.inf
    max_running_episode_reward = -np.inf
    best_model = None
    LOGGER.info("Initializing learner ------")
    device = get_device()
    ppo_learner = PPO(cfgs, device, ActorCriticPolicy)
    if args.load and glob.glob(cfgs.get("model_path") + "/*.pth"):
        LOGGER.info("Loading model from %s", cfgs.get("model_path"))
        ppo_learner.load(cfgs.get("model_path"), suffix=args.suffix, load_all=True)

    # One RolloutBuffer holds one agent's whole-episode trajectory.
    trajectories = [
        RolloutBuffer(cfgs.getfloat("gae_lambda", 1), cfgs.getfloat("gamma", 0.99))
        for _ in range(env.numAgents)
    ]
    batch_size = cfgs.getint("batch_size")

    while episode < args.num_episodes:
        env.reset(scattered=True, fixed_map=args.fixed_map)
        done = False
        episode_reward = 0
        episode_onNode = np.zeros(env.numAgents)
        next_obs_dict = defaultdict(list)
        for _ in range(env._max_episode_steps):
            if args.render:
                env.render(True, True)

            # If no agent is on a node, everyone just coasts one step.
            if not env.checkAnyOnNode():
                for id in range(1, env.numAgents + 1):
                    act = env.agents[id - 1].next_target_ind
                    env.agents[id - 1].move(act)
                env.tick()
                continue

            act_list = []
            obs_list = []
            obs_tensor_list = []
            value_list = []
            act_log_prob_list = []
            reward_list = []
            onNode_agents = []
            onEdge_agents = [i + 1 for i in range(env.numAgents)]

            for id in range(1, env.numAgents + 1):
                if env.agents[id - 1].isOnNode:
                    env.communicate(id)
            env.updateEnv()

            for id in range(1, env.numAgents + 1):
                if env.agents[id - 1].isOnNode:
                    onNode_agents.append(id)
                    obs = env.observe(id, relative=True)
                    obs_tensor = [torch.FloatTensor(item).to(device) for item in obs]
                    obs_tensor_list.append(obs_tensor)
                    obs_list.append(obs)

            ag_states = torch.cat([ag for ag, _, _, _ in obs_tensor_list], axis=0)
            tg_states = torch.cat([tg for _, tg, _, _ in obs_tensor_list], axis=0)
            cn_states = torch.cat([cn for _, _, cn, _ in obs_tensor_list], axis=0)
            mask_states = torch.cat([mk for _, _, _, mk in obs_tensor_list], axis=0)
            obs_batch = (ag_states, tg_states, cn_states, mask_states)

            with torch.no_grad():
                act_tensor, act_log_prob_tensor, value_tensor = ppo_learner.act(obs_batch)
                act_array = act_tensor.cpu().numpy()
                act_log_prob_array = act_log_prob_tensor.cpu().numpy()
                value_array = value_tensor.cpu().numpy()
                act_list.append(act_array)
                act_log_prob_list.append(act_log_prob_array)
                value_list.append(value_array)

            for i, id in enumerate(onNode_agents):
                act = act_array[i].item()
                reward = env.get_agent_reward(id, localReward=False)
                env.agents[id - 1].move(act)
                next_obs = env.observe(id)
                next_obs_dict[id - 1].append(next_obs)
                episode_reward += reward
                reward_list.append(reward)

            for i in onNode_agents:
                onEdge_agents.remove(i)
            for id in onEdge_agents:
                act = env.agents[id - 1].next_target_ind
                env.agents[id - 1].move(act)
            done = env.checkFinished()

            for i, id in enumerate(onNode_agents):
                trajectories[id - 1].add(
                    obs_list[i],
                    act_list[0][i],
                    reward_list[i],
                    done,
                    value_list[0][i],
                    act_log_prob_list[0][i],
                )

            if done:
                break
            global_step += 1

            step_onNode_indice = [i - 1 for i in onNode_agents]
            episode_onNode[step_onNode_indice] = 1

        # End of episode: only post-process agents that visited a node.
        episode_onNode_indice = list(np.array(np.where(episode_onNode == 1))[0])
        episode_valid_indice = []
        if len(episode_onNode_indice) == 0:
            continue

        for idx in episode_onNode_indice:
            if len(trajectories[idx].get()) > 1:
                episode_valid_indice.append(idx)
                # Shift rewards one step ahead, then drop the two trailing steps.
                last = trajectories[idx].get()[-1]
                trajectories[idx].add(
                    last.obs, last.action, last.reward, last.done, last.value, last.log_prob
                )
                data_list = trajectories[idx].get()
                for j in range(0, len(data_list) - 1):
                    data_list[j].reward = data_list[j + 1].reward
                trajectories[idx].pop(2)
            else:
                trajectories[idx].pop(1)

        trajectory_list = []
        with torch.no_grad():
            for idx in episode_valid_indice:
                buffer = trajectories[idx]
                next_obs_tensor = [
                    torch.FloatTensor(item).to(device) for item in next_obs_dict[idx][-1]
                ]
                _, _, last_v = ppo_learner.act(tuple(next_obs_tensor))
                buffer.post_process(last_v.item())
                trajectory_list.append(buffer.get_batch())
                buffer.reset()
        episode += 1
        episode_rewards.append(episode_reward)
        running_episode_reward = np.mean(episode_rewards)
        if max_episode_reward < episode_reward:
            max_episode_reward = episode_reward
        if max_running_episode_reward < running_episode_reward:
            best_model = deepcopy(ppo_learner.get_weights())
            LOGGER.info(
                "[Episode %d|%d] new best running reward: %.4f (prev %s)",
                episode,
                args.num_episodes,
                running_episode_reward,
                max_running_episode_reward,
            )
            max_running_episode_reward = running_episode_reward

        if not episode_valid_indice:
            continue

        v_l = pg_l = ent_l = clip_frac = approx_kl_div = 0.0
        for batch in make_batch(batch_size, trajectory_list):
            v_l, pg_l, ent_l, clip_frac, approx_kl_div = ppo_learner.update(batch)

        if writer is not None:
            writer.add_scalar("Perf/episode_reward", episode_reward, global_step=episode)
            writer.add_scalar(
                "Perf/running_episode_reward", running_episode_reward, global_step=episode
            )
            writer.add_scalar("Train/value_loss", v_l, global_step=episode)
            writer.add_scalar("Train/policy_loss", pg_l, global_step=episode)
            writer.add_scalar("Train/entropy_loss", ent_l, global_step=episode)
            writer.add_scalar("Train/clip_fraction", clip_frac, global_step=episode)
            writer.add_scalar("Train/approx_kl_div", approx_kl_div, global_step=episode)
            writer.flush()
        LOGGER.info(
            "[Episode %d] value loss: %.4f, policy loss: %.4f, episode reward: %.4f, running: %.4f",
            episode,
            v_l,
            pg_l,
            episode_reward,
            running_episode_reward,
        )

        if episode % args.episode_segment == 0:
            phase += 1
            LOGGER.info("Phase training finished, saving best model ...")
            ppo_learner.set_weights(best_model)
            ppo_learner.save(cfgs.get("model_path"), phase)
            ppo_learner.lr *= 0.1

    LOGGER.info("Training finished, saving best model ...")
    ppo_learner.set_weights(best_model, load_all=True)
    ppo_learner.save(cfgs.get("model_path"), args.suffix)


def _evaluate(args, cfgs, env):
    test_device = get_device(force_cpu=True)
    LOGGER.info("Initializing tester ------")
    tester = PPO(cfgs, test_device, ActorCriticPolicy)
    if glob.glob(cfgs.get("model_path") + "/*.pth"):
        LOGGER.info("Loading model from %s", cfgs.get("model_path"))
        tester.load(cfgs.get("model_path"), args.suffix)
    test_path = cfgs.get("test_path")
    for eps in range(args.num_tests):
        env.reset(scattered=True, fixed_map=args.fixed_map)
        done = False
        score = 0
        for _ in range(env._max_episode_steps):
            obs_tensor_list = []
            onNode_agents = []
            onEdge_agents = [i + 1 for i in range(env.numAgents)]

            env.render(True, True)
            if done:
                env.close()
                break

            for id in range(1, env.numAgents + 1):
                if env.agents[id - 1].isOnNode:
                    onNode_agents.append(id)
                    env.communicate(id)
            env.updateEnv()

            for i in onNode_agents:
                onEdge_agents.remove(i)

            for id in range(1, env.numAgents + 1):
                if env.agents[id - 1].isOnNode:
                    obs = env.observe(id, relative=True)
                    obs_tensor = tuple(torch.FloatTensor(item).to(test_device) for item in obs)
                    obs_tensor_list.append(obs_tensor)

            with torch.no_grad():
                for i, id in enumerate(onNode_agents):
                    act_tensor, _, _ = tester.act(obs_tensor_list[i])
                    act = act_tensor.cpu().squeeze().numpy().item()
                    reward = env.get_agent_reward(id, localReward=False)
                    env.agents[id - 1].move(act)
                    score += reward
                for id in onEdge_agents:
                    act = env.agents[id - 1].next_target_ind
                    env.agents[id - 1].move(act)

            done = env.checkFinished()
            score = score / env.numAgents
        LOGGER.info("[Test episode-%d] Test perf: %s", eps, score)
        comms = env.comms_radius if env.comms_radius is not None else "inf"
        env.save(
            f"{test_path}/agents{env.numAgents}-targets{env.numTargets}"
            f"-comms_radius{comms}-test{eps}.gif"
        )
