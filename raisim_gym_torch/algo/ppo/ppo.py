from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage import RolloutStorage


class PPO:
    def __init__(self,
                 actor_critic_module,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 learning_rate_schedule='adaptive',
                 desired_kl=0.01,
                 decay_gamma=1,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 learning_rate_min=None):

        # PPO components
        self.actor_critic_module = actor_critic_module
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, self.actor_critic_module.obs_shape,
                                      self.actor_critic_module.action_shape, device)

        if self.actor_critic_module.shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam(self.actor_critic_module.parameters(), lr=learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # ADAM
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay_gamma)
        self.learning_rate_min = learning_rate_min

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.obs = None
        self.values = None

    def act(self, obs):
        self.obs = obs
        obs = torch.from_numpy(self.obs).to(self.device)

        self.actions, self.actions_log_prob = self.actor_critic_module.sample(obs)

        if self.actor_critic_module.predict_values_during_act:
            self.values = self.actor_critic_module.predict(obs).cpu().numpy()
        else:
            self.values = None

        return self.actions

    def step(self, rews, dones):
        self.actor_critic_module.update_dones(dones)

        self.storage.add_transitions(self.obs, self.actions, self.actor_critic_module.action_mean,
                                     self.actor_critic_module.distribution.std_np, rews, self.values, dones,
                                     self.actions_log_prob)

    def update(self, obs, log_this_iteration, update):
        last_values = self.actor_critic_module.predict(torch.from_numpy(obs).to(self.device))

        if not self.actor_critic_module.predict_values_during_act:
            self.values = self.actor_critic_module.predict(
                torch.from_numpy(self.storage.obs).to(self.device)).cpu().numpy()
        else:
            self.values = None

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.values, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, infos = self._train_step(log_this_iteration, update)

        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor_critic_module.distribution.std.mean()
        self.writer.add_scalar('PPO/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('PPO/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('PPO/mean_noise_std', mean_std.item(), variables['it'])
        self.writer.add_scalar('PPO/learning_rate', self.learning_rate, variables['it'])

    def _train_step(self, log_this_iteration, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        backward_count = 0

        for epoch in range(self.num_learning_epochs):
            if epoch == 0 and getattr(self.actor_critic_module, 'full_batch_evaluation', False) and getattr(
                    self.actor_critic_module, 'synchronous_update', True):
                num_mini_batches = 1
            else:
                num_mini_batches = self.num_mini_batches

            for obs_batch, actions_batch, old_sigma_batch, old_mu_batch, current_values_batch, advantages_batch, \
                returns_batch, old_actions_log_prob_batch, indices in self.batch_sampler(num_mini_batches):

                with torch.autograd.set_detect_anomaly(False):
                    actions_log_prob_batch, entropy_batch, value_batch = self.actor_critic_module.evaluate(
                        obs_batch, actions_batch, indices)

                    # Adjusting the learning rate using KL divergence
                    mu_batch = self.actor_critic_module.action_mean
                    sigma_batch = self.actor_critic_module.distribution.std

                    # KL
                    if self.desired_kl is not None and self.schedule == 'adaptive':
                        with torch.no_grad():
                            kl = torch.sum(
                                torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                        torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                        2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                            kl_mean = torch.mean(kl)

                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                            elif self.desired_kl / 2.0 > kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.2)

                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.learning_rate

                    # Surrogate loss
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                       1.0 + self.clip_param)
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    # Value function loss
                    if self.use_clipped_value_loss:
                        value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(
                            -self.clip_param,
                            self.clip_param)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()

                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                    if hasattr(self.actor_critic_module, 'loss'):
                        actor_critic_module_loss = self.actor_critic_module.loss(
                            writer=self.writer if log_this_iteration and backward_count == 0 else None, it=it)

                        if actor_critic_module_loss is not None:
                            loss += actor_critic_module_loss

                    # Gradient step
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    backward_count += 1

                    if log_this_iteration:
                        mean_value_loss += value_loss.item()
                        mean_surrogate_loss += surrogate_loss.item()

        if self.schedule == 'decay':
            if self.learning_rate_min is None or self.optimizer.param_groups[0]["lr"] > self.learning_rate_min:
                self.scheduler.step()

            self.learning_rate = self.optimizer.param_groups[0]["lr"]

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, locals()
