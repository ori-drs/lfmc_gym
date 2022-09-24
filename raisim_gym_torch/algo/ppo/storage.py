import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, action_shape, device, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = device

        # Core
        self.obs = np.zeros([num_transitions_per_env, num_envs, *obs_shape], dtype=np.float32)
        self.rewards = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.actions = np.zeros([num_transitions_per_env, num_envs, *action_shape], dtype=np.float32)
        self.dones = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.bool)

        # For PPO
        self.actions_log_prob = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.values = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.returns = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.advantages = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.mu = np.zeros([num_transitions_per_env, num_envs, *action_shape], dtype=np.float32)
        self.sigma = np.zeros([num_transitions_per_env, num_envs, *action_shape], dtype=np.float32)

        # torch variables
        self.obs_tc = torch.from_numpy(self.obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_tc = torch.from_numpy(self.advantages).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

    def add_transitions(self, obs, actions, mu, sigma, rewards, values, dones, actions_log_prob):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.mu[self.step] = mu
        self.sigma[self.step] = sigma
        self.rewards[self.step] = rewards.reshape(-1, 1)

        if values is not None:
            self.values[self.step] = values.reshape(-1, 1)

        self.dones[self.step] = dones.reshape(-1, 1)
        self.actions_log_prob[self.step] = actions_log_prob.reshape(-1, 1)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, values, gamma, lam):
        if values is not None:
            self.values = values

        advantage = 0

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values.cpu().numpy()
                # next_is_not_terminal = 1.0 - self.dones[step].float()
            else:
                next_values = self.values[step + 1]
                # next_is_not_terminal = 1.0 - self.dones[step+1].float()

            next_is_not_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Convert to torch variables
        self.obs_tc = torch.from_numpy(self.obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_tc = torch.from_numpy(self.advantages).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            obs_batch = self.obs_tc.view(-1, *self.obs_tc.size()[2:])[indices]
            actions_batch = self.actions_tc.view(-1, self.actions_tc.size(-1))[indices]
            sigma_batch = self.sigma_tc.view(-1, self.sigma_tc.size(-1))[indices]
            mu_batch = self.mu_tc.view(-1, self.mu_tc.size(-1))[indices]
            values_batch = self.values_tc.view(-1, 1)[indices]
            returns_batch = self.returns_tc.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.actions_log_prob_tc.view(-1, 1)[indices]
            advantages_batch = self.advantages_tc.view(-1, 1)[indices]
            yield obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, \
                  old_actions_log_prob_batch, indices

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        transition_range = mini_batch_size // self.num_envs

        if transition_range < 1:
            raise RuntimeError

        obs_size = self.obs_tc.size()
        action_dim = self.actions_tc.size(-1)
        sigma_dim = self.sigma_tc.size(-1)
        mu_dim = self.mu_tc.size(-1)

        for batch_id in range(num_mini_batches):
            yield self.obs_tc.view(-1, self.num_envs, *obs_size[2:])[
                  batch_id * transition_range:(batch_id + 1) * transition_range].view(-1, *obs_size[2:]), \
                  self.actions_tc.view(-1, self.num_envs, action_dim)[
                  batch_id * transition_range:(batch_id + 1) * transition_range].view(-1, action_dim), \
                  self.sigma_tc.view(-1, self.num_envs, sigma_dim)[
                  batch_id * transition_range:(batch_id + 1) * transition_range].view(-1, sigma_dim), \
                  self.mu_tc.view(-1, self.num_envs, mu_dim)[
                  batch_id * transition_range:(batch_id + 1) * transition_range].view(-1, mu_dim), \
                  self.values_tc.view(-1, self.num_envs, 1)[
                  batch_id * transition_range:(batch_id + 1) * transition_range].view(-1, 1), \
                  self.advantages_tc.view(-1, self.num_envs, 1)[
                  batch_id * transition_range:(batch_id + 1) * transition_range].view(-1, 1), \
                  self.returns_tc.view(-1, self.num_envs, 1)[
                  batch_id * transition_range:(batch_id + 1) * transition_range].view(-1, 1), \
                  self.actions_log_prob_tc.view(-1, self.num_envs, 1)[
                  batch_id * transition_range:(batch_id + 1) * transition_range].view(-1, 1), \
                  [batch_id * transition_range, (batch_id + 1) * transition_range]
