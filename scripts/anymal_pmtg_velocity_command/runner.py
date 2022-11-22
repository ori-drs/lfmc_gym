import os
import psutil

import math
import time
import argparse
import numpy as np

import torch
import random

from ruamel.yaml import YAML, dump, RoundTripDumper

from raisim_gym_torch.env.bin.anymal_pmtg_velocity_command import RaisimGymEnv
from raisim_gym_torch.env.bin.anymal_pmtg_velocity_command import NormalSampler

import modules

from raisim_gym_torch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisim_gym_torch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher, RewardLogger

import raisim_gym_torch.algo.ppo.ppo as PPO


def main():
    # task specification
    task_name = "anymal_pmtg_velocity_command"

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
    parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
    args = parser.parse_args()
    mode = args.mode
    weight_path = args.weight

    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directories
    home_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../../')
    task_path = home_path + "/gym_envs/" + task_name

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

    if cfg['environment']['num_threads'] == 'auto':
        cfg['environment']['num_threads'] = psutil.cpu_count(logical=False)

    # create environment from the configuration file
    env = VecEnv(RaisimGymEnv(home_path + "/resources", dump(cfg['environment'], Dumper=RoundTripDumper)))

    # Set seed
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    env.seed(cfg['seed'])

    # Reward logger to compute the means and stds of individual reward terms
    reward_logger = RewardLogger(env, cfg, episodes_window=100)

    # Training
    n_steps = cfg['algorithm']['update_steps']
    total_steps = n_steps * env.num_envs

    last_ppo_log_iter = -cfg['environment']['log_interval']

    _actor_critic_module = modules.get_actor_critic_module_from_config(cfg, env, NormalSampler, device)
    actor_critic_module_eval = modules.get_actor_critic_module_from_config(cfg, env, NormalSampler, device)

    saver = ConfigurationSaver(log_dir=home_path + "/data/" + task_name,
                               save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
    tensorboard_launcher(saver.data_dir)  # press refresh (F5) after the first ppo update

    # Discount factor
    rl_gamma = np.exp(cfg['environment']['control_dt'] * np.log(0.5) / cfg['algorithm']['gamma_half_life_duration'])
    learning_rate_decay_gamma = np.exp(np.log(
        cfg['algorithm']['learning_rate']['final'] / cfg['algorithm']['learning_rate']['initial']
    ) / cfg['algorithm']['learning_rate']['decay_steps'])  # x_t = x_0 * gamma ^ t

    if cfg['algorithm']['learning_rate']['mode'] == 'constant':
        learning_rate_decay_gamma = 1.

    ppo = PPO.PPO(
        actor_critic_module=_actor_critic_module,
        num_envs=cfg['environment']['num_envs'],
        num_transitions_per_env=n_steps,
        num_learning_epochs=4,
        gamma=rl_gamma,
        lam=0.95,
        num_mini_batches=4,
        device=device,
        log_dir=saver.data_dir,
        learning_rate=cfg['algorithm']['learning_rate']['initial'],
        entropy_coef=0.0,
        learning_rate_schedule=cfg['algorithm']['learning_rate']['mode'],
        learning_rate_min=cfg['algorithm']['learning_rate']['min'],
        decay_gamma=learning_rate_decay_gamma
    )

    if mode == 'retrain':
        load_param(weight_path, env, ppo.actor_critic_module, ppo.optimizer, saver.data_dir)

    for update in range(20500):
        start = time.time()

        # If true, only those environments which meet a condition are reset - for example, if max episode length
        # is not reached, the environment will not be reset.
        reset_indices = env.reset(conditional_reset=update != 0)
        ppo.actor_critic_module.reset()

        env.set_max_episode_length(cfg['environment']['max_time'])
        env.enable_early_termination()

        reward_ll_sum = 0
        done_sum = 0

        visualizable_iteration = False

        if update % cfg['environment']['save_every_n'] == 0:
            print('Storing Actor Critic Module Parameters')
            parameters_save_dict = {'optimizer_state_dict': ppo.optimizer.state_dict()}
            ppo.actor_critic_module.save_parameters(saver.data_dir + "/full_" + str(update) + '.pt',
                                                    parameters_save_dict)
            env.save_scaling(saver.data_dir, str(update))

        if update % cfg['environment']['eval_every_n'] == 0:
            if cfg['environment']['render']:
                visualizable_iteration = True
                print('Visualizing and Evaluating the Current Policy')

                # we create another graph just to demonstrate the save/load method
                actor_critic_module_eval.load_parameters(saver.data_dir + "/full_" + str(update) + '.pt')

                env.turn_on_visualization()

                if cfg['record_video']:
                    env.start_video_recording(saver.data_dir + "/policy_" + str(update) + '.mp4')

                for step in range(math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])):
                    with torch.no_grad():
                        frame_start = time.time()
                        obs = env.observe(False)

                        action_ll = actor_critic_module_eval.generate_action(
                            torch.from_numpy(obs).to(device)).cpu().detach().numpy()

                        reward_ll, dones = env.step(action_ll)
                        actor_critic_module_eval.update_dones(dones)
                        frame_end = time.time()
                        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
                        if wait_time > 0.:
                            time.sleep(wait_time)

                if cfg['record_video']:
                    env.stop_video_recording()

                env.turn_off_visualization()

                env.reset()
                ppo.actor_critic_module.reset()

        # actual training
        for step in range(n_steps):
            obs = env.observe()
            action = ppo.act(obs)
            reward, dones = env.step(action)
            ppo.step(rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)

            # Store the rewards for this step
            reward_logger.step()

        # take st step to get value obs
        obs = env.observe()

        log_this_iteration = update % cfg['environment']['log_interval'] == 0

        if not visualizable_iteration:
            if update - last_ppo_log_iter > cfg['environment']['log_interval']:
                log_this_iteration = True

            if log_this_iteration:
                last_ppo_log_iter = update

            ppo.update(obs=obs, log_this_iteration=log_this_iteration, update=update)

            ppo.actor_critic_module.update()
            ppo.actor_critic_module.distribution.enforce_minimum_std((torch.ones(16) * 0.2).to(device))

            # curriculum update. Implement it in Environment.hpp
            env.curriculum_callback()

        ppo.storage.clear()

        average_ll_performance = reward_ll_sum / total_steps
        average_dones = done_sum / total_steps

        # Add to tensorboard
        if log_this_iteration and not visualizable_iteration:
            ppo.writer.add_scalar('Rewards/Episodic/average_ll', average_ll_performance, update)
            reward_logger.log_to_tensorboard(ppo.writer, update)

        # Clear the episodic reward storage buffer
        reward_logger.episodic_reset()

        end = time.time()

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                           * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')


if __name__ == '__main__':
    main()
