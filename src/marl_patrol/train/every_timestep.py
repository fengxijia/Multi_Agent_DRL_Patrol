"""Training / evaluation loop for the *every-timestep* decision paradigm.

Every agent acts at every simulation step through ``env.step``. The PPO update
and reward logic are unchanged from the original ``MARL_Patrol_EveryTimeStep``
script; imports, device handling and batching were refactored, and a couple of
crashes in the original evaluation branch were fixed (see ``_evaluate``).
"""

import glob
from collections import deque
from copy import deepcopy

import numpy as np
import torch

from marl_patrol.algorithms.ppo import PPO
from marl_patrol.buffers.rollout_buffer import RolloutBuffer
from marl_patrol.envs.every_timestep import MAPatrolEnv
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
    episode_rewards = deque(maxlen=cfgs.getint("window_size"))
    max_episode_reward = -np.inf
    max_running_episode_reward = -np.inf
    best_model = None
    LOGGER.info("Initializing learner ------")
    device = get_device()
    ppo_learner = PPO(cfgs, device, ActorCriticPolicy)
    if args.load and glob.glob(cfgs.get("model_path") + "/*.pth"):
        LOGGER.info("Loading model from %s", cfgs.get("model_path"))
        ppo_learner.load(cfgs.get("model_path"), args.suffix)

    trajectories = [
        RolloutBuffer(cfgs.getfloat("gae_lambda", 1), cfgs.getfloat("gamma", 0.99))
        for _ in range(env.numAgents)
    ]
    batch_size = cfgs.getint("batch_size")

    while episode < args.num_episodes:
        env.reset(scattered=True)
        done = False
        episode_reward = 0
        next_obs_list = []
        for _ in range(env._max_episode_steps):
            act_list = []
            obs_list = []
            obs_tensor_list = []
            next_obs_list = []
            value_list = []
            act_log_prob_list = []
            reward_list = []

            for id in range(1, env.numAgents + 1):
                env.communicate(id)
            env.updateEnv()

            for id in range(1, env.numAgents + 1):
                obs = env.observe(id, relative=True)
                obs_tensor = tuple(torch.FloatTensor(item).to(device) for item in obs)
                obs_tensor_list.append(obs_tensor)
                obs_list.append(obs)

            for id in range(1, env.numAgents + 1):
                with torch.no_grad():
                    act_tensor, act_log_prob_tensor, value_tensor = ppo_learner.act(
                        obs_tensor_list[id - 1]
                    )
                    act_list.append(act_tensor.cpu().numpy())
                    act_log_prob_list.append(act_log_prob_tensor.cpu().numpy())
                    value_list.append(value_tensor.item())

            for id in range(1, env.numAgents + 1):
                act = act_list[id - 1].squeeze(0)[0]
                agent_states, target_states, connectivity_feature, local_mask, _, _, reward = (
                    env.step(id, act)
                )
                next_obs_list.append(
                    (agent_states, target_states, connectivity_feature, local_mask)
                )
                episode_reward += reward
                reward_list.append(reward)

            done = env.checkFinished()

            for idx, buffer in enumerate(trajectories):
                buffer.add(
                    obs_list[idx],
                    act_list[idx],
                    reward_list[idx],
                    done,
                    value_list[idx],
                    act_log_prob_list[idx],
                )

            if done:
                break
            global_step += 1

        # Bootstrap value and post-process every agent's trajectory.
        trajectory_list = []
        with torch.no_grad():
            for idx, buffer in enumerate(trajectories):
                next_obs_tensor = tuple(
                    torch.FloatTensor(item).to(device) for item in next_obs_list[idx]
                )
                _, _, last_v = ppo_learner.act(next_obs_tensor)
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
            "[Episode %d] value loss: %.4f, policy loss: %.4f, episode reward: %s, running: %.4f",
            episode,
            v_l,
            pg_l,
            episode_reward,
            running_episode_reward,
        )

    LOGGER.info("Training finished, saving best model ...")
    ppo_learner.set_weights(best_model)
    ppo_learner.save(cfgs.get("model_path"), args.suffix)


def _evaluate(args, cfgs, env):
    test_device = get_device(force_cpu=True)
    LOGGER.info("Initializing tester ------")
    tester = PPO(cfgs, test_device, ActorCriticPolicy)
    if glob.glob(cfgs.get("model_path") + "/*.pth"):
        LOGGER.info("Loading model from %s", cfgs.get("model_path"))
        tester.load(cfgs.get("model_path"), args.suffix)
    test_path = cfgs.get("test_path")
    for i in range(args.num_tests):
        env.reset(scattered=True)
        done = False
        score = 0
        for _ in range(env._max_episode_steps):
            act_list = []
            obs_list = []
            obs_tensor_list = []  # BUGFIX: was never initialised in the original eval loop.
            if args.render:
                env.render(True, True)
            if done:
                env.close()
                break
            for id in range(1, env.numAgents + 1):
                env.communicate(id)
            env.updateEnv()
            for id in range(1, env.numAgents + 1):
                obs = env.observe(id, relative=True)
                # BUGFIX: use test_device (the original referenced an undefined ``device``).
                obs_tensor = tuple(torch.FloatTensor(item).to(test_device) for item in obs)
                obs_tensor_list.append(obs_tensor)
                obs_list.append(obs)
            with torch.no_grad():
                for id in range(1, env.numAgents + 1):
                    act_tensor, _, _ = tester.act(obs_tensor_list[id - 1])
                    act_list.append(act_tensor.cpu().numpy())
            for id in range(1, env.numAgents + 1):
                act = act_list[id - 1].squeeze(0)[0]
                _, _, _, _, _, _, reward = env.step(id, act)
                score += reward
            done = env.checkFinished()
            score = score / env.numAgents
        LOGGER.info("[Test episode-%d] Test perf: %s", i, score)
        comms = env.comms_radius if env.comms_radius is not None else "inf"
        env.save(
            f"{test_path}/agents{env.numAgents}-targets{env.numTargets}"
            f"-comms_radius{comms}-max_steps{env.max_episode_steps}.gif"
        )
