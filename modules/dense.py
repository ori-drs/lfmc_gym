import torch
import numpy as np

from modules.common import ActorNetworkArchitecture, CriticNetworkArchitecture

import raisim_gym_torch.algo.ppo.module as module
import raisim_gym_torch.algo.ppo.distributions as distributions


class DenseActorCriticModule(module.ActorCriticModule):
    def __init__(self, obs_shape, action_shape, actor_hidden_layers, actor_activation, critic_hidden_layers,
                 critic_activation, action_sampler, seed, num_envs=1, device='cpu', shuffle_batch=True,
                 predict_values_during_act=False, initial_action_std=1.0, compute_jacobian=False,
                 observation_indices=None, network_weights_gain=np.sqrt(2)):
        if observation_indices is None:
            self._actor_input_indices = list(range(obs_shape))
            self._critic_input_indices = list(range(obs_shape))
        else:
            self._actor_input_indices = list(
                range(observation_indices['actor_input'][0], observation_indices['actor_input'][1]))
            self._critic_input_indices = list(
                range(observation_indices['critic_input'][0], observation_indices['critic_input'][1]))

        # Create actor and critic network architectures
        actor_architecture = ActorNetworkArchitecture(input_dim=len(self._actor_input_indices),
                                                      hidden_layers=actor_hidden_layers,
                                                      output_dim=action_shape, activation=actor_activation,
                                                      network_weights_gain=network_weights_gain)
        critic_architecture = CriticNetworkArchitecture(input_dim=len(self._critic_input_indices),
                                                        hidden_layers=critic_hidden_layers,
                                                        activation=critic_activation,
                                                        network_weights_gain=network_weights_gain)

        actor = module.Actor(actor_architecture, distributions.MultivariateGaussianDiagonalCovariance(
            action_shape, num_envs, initial_action_std, action_sampler(action_shape), seed), device)
        critic = module.Critic(critic_architecture, device)

        # Initialize base constructor
        super().__init__(actor, critic, shuffle_batch, predict_values_during_act)

        self._evaluate_count = 0
        self._compute_jacobian = compute_jacobian

        self._jacobian = None

    def evaluate(self, obs, actions, indices):
        actions_log_prob, entropy, value = super().evaluate(obs[:, self._actor_input_indices], actions, indices)

        # Only compute network Jacobian during the first evaluation before reset
        if self._compute_jacobian and self._evaluate_count == 0:
            batch_size = min(100, obs.shape[0])

            # Indexing at the end gets rid of all the 0s for inputs that don't correspond to outputs
            self._jacobian = torch.autograd.functional.jacobian(
                self._actor.architecture.forward, obs[-batch_size:, self._actor_input_indices], vectorize=True,
                strategy='reverse-mode')[torch.arange(batch_size), :, torch.arange(batch_size), :]

        self._evaluate_count += 1
        return actions_log_prob, entropy, value

    def sample(self, obs):
        return super().sample(obs[:, self._actor_input_indices])

    def generate_action(self, obs):
        return super().generate_action(obs[:, self._actor_input_indices])

    def predict(self, obs):
        return super().predict(obs[..., self._critic_input_indices])

    def reset(self):
        self._evaluate_count = 0
        self._jacobian = None

    def loss(self, writer=None, it=0):
        if writer is None or self._jacobian is None:
            return None

        jacobian_mean, jacobian_std = torch.std_mean(torch.abs(self._jacobian), dim=0)

        jacobian_mean = jacobian_mean / torch.max(jacobian_mean)
        jacobian_std = jacobian_std / torch.max(jacobian_std)

        writer.add_image('NetworkJacobian/Mean', jacobian_mean.transpose(0, 1), it, dataformats='HW')
        writer.add_image('NetworkJacobian/Std', jacobian_std.transpose(0, 1), it, dataformats='HW')

        return None
