from shutil import copyfile
import datetime
import os
import ntpath
import numpy as np
from collections import deque


class ConfigurationSaver:
    def __init__(self, log_dir, save_items):
        self._data_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir


def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: " + url)
    webbrowser.open_new(url)


class RewardLogger:
    def __init__(self, env, cfg, episodes_window=10):
        self._env = env

        # Get the reward terms for plotting
        self._reward_terms = [key for key in cfg['environment']['reward'].keys()]

        # For episodic rewards
        self._reward_storage_dict = dict()
        self._reward_mean_dict = dict()
        self._reward_std_dict = dict()

        # For a window of episodic rewards
        self._reward_mean_moving_storage_dict = dict()
        self._reward_std_moving_storage_dict = dict()

        self._reward_mean_of_means_dict = dict()
        self._reward_std_of_means_dict = dict()

        self._reward_mean_of_stds_dict = dict()
        self._reward_std_of_stds_dict = dict()

        self._episodes_window = episodes_window
        self._episodic_reset_count = 0

        self.reset()

    def step(self):
        # Log rewards
        envs_reward_info = self._env.get_reward_info()

        for reward_info in envs_reward_info:
            for term in self._reward_terms:
                self._reward_storage_dict[term].append(reward_info[term])

    def log_to_tensorboard(self, writer, it):
        # Compute Episodic Mean and Standard Deviation of Individual Reward Terms
        for term in self._reward_terms:
            self._reward_mean_dict[term] = np.mean(np.array(self._reward_storage_dict[term]))
            self._reward_std_dict[term] = np.std(np.array(self._reward_storage_dict[term]))

            if self._episodic_reset_count > 0:
                self._reward_mean_of_means_dict[term] = np.mean(np.array(self._reward_mean_moving_storage_dict[term]))
                self._reward_std_of_means_dict[term] = np.std(np.array(self._reward_mean_moving_storage_dict[term]))

                self._reward_mean_of_stds_dict[term] = np.mean(np.array(self._reward_std_moving_storage_dict[term]))
                self._reward_std_of_stds_dict[term] = np.std(np.array(self._reward_std_moving_storage_dict[term]))

        writer.add_scalars('Rewards/Episodic/mean', self._reward_mean_dict, it)
        writer.add_scalars('Rewards/Episodic/std', self._reward_std_dict, it)

        if self._episodic_reset_count > 0:
            writer.add_scalars('Rewards/Windowed/Means/mean', self._reward_mean_of_means_dict, it)
            writer.add_scalars('Rewards/Windowed/Means/std', self._reward_std_of_means_dict, it)

            writer.add_scalars('Rewards/Windowed/Stds/mean', self._reward_mean_of_stds_dict, it)
            writer.add_scalars('Rewards/Windowed/Stds/std', self._reward_std_of_stds_dict, it)

    def episodic_reset(self):
        # Reset episodic storage dict and add the episodic mean and std to the moving dicts
        for term in self._reward_terms:
            self._reward_storage_dict[term] = []

            self._reward_mean_moving_storage_dict[term].append(self._reward_mean_dict[term])
            self._reward_std_moving_storage_dict[term].append(self._reward_std_dict[term])

        self._episodic_reset_count += 1

    def reset(self):
        for term in self._reward_terms:
            self._reward_storage_dict[term] = []
            self._reward_mean_dict[term] = 0
            self._reward_std_dict[term] = 0

            self._reward_mean_moving_storage_dict[term] = deque(maxlen=self._episodes_window)
            self._reward_std_moving_storage_dict[term] = deque(maxlen=self._episodes_window)

            self._reward_mean_of_means_dict[term] = 0
            self._reward_std_of_means_dict[term] = 0

            self._reward_mean_of_stds_dict[term] = 0
            self._reward_std_of_stds_dict[term] = 0

            self._episodic_reset_count = 0


def load_param(weight_path, env, actor_critic_module, optimizer, data_dir):
    if weight_path == "":
        raise Exception(
            "\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the checkpoint:", weight_path + "\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    mean_csv_path = weight_dir + 'mean' + iteration_number + '.csv'
    var_csv_path = weight_dir + 'var' + iteration_number + '.csv'
    items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir + "cfg.yaml", weight_dir + "Environment.hpp"]

    if items_to_save is not None:
        pretrained_data_dir = data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]
        os.makedirs(pretrained_data_dir)
        for item_to_save in items_to_save:
            copyfile(item_to_save, pretrained_data_dir + '/' + item_to_save.rsplit('/', 1)[1])

    # load observation scaling from files of pre-trained model
    env.load_scaling(weight_dir, iteration_number)

    # load actor and critic parameters from full checkpoint
    checkpoint = actor_critic_module.load_parameters(weight_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
