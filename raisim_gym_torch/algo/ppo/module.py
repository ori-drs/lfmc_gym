import torch
import torch.nn as nn

from .networks import NetworkArchitecture


class Actor(nn.Module):
    def __init__(self, architecture: NetworkArchitecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None

        self.action_tensor = None

    def sample(self, obs):
        self.action_tensor = self.architecture.forward(obs)
        self.action_mean = self.action_tensor.cpu().numpy()
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture.forward(obs)
        return self.distribution.evaluate(self.action_mean, actions)

    def noiseless_action(self, obs):
        return self.architecture.forward(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic(nn.Module):
    def __init__(self, architecture: NetworkArchitecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.forward(obs).detach()

    def evaluate(self, obs):
        return self.architecture.forward(obs)

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class ActorCriticModule(nn.Module):
    def __init__(self, actor: Actor, critic: Critic, shuffle_batch=False, predict_values_during_act=False):
        super().__init__()

        self._actor = actor
        self._critic = critic

        self._shuffle_batch = shuffle_batch
        self._predict_values_during_act = predict_values_during_act

    def sample(self, obs):
        with torch.no_grad():
            return self._actor.sample(obs)

    def predict(self, obs):
        with torch.no_grad():
            return self._critic.predict(obs)

    def evaluate(self, obs, actions, indices):
        actions_log_prob, entropy = self._actor.evaluate(obs, actions)
        value = self._critic.evaluate(obs)
        return actions_log_prob, entropy, value

    def update(self):
        self._actor.update()

    def update_dones(self, dones):
        pass

    def reset(self):
        pass

    def generate_action(self, obs):
        with torch.no_grad():
            return self._actor.architecture.forward(obs)

    def save_parameters(self, save_path, parameters_dict=None):
        if parameters_dict is None:
            parameters_dict = dict()

        parameters_dict['actor_architecture_state_dict'] = self._actor.architecture.state_dict()
        parameters_dict['actor_distribution_state_dict'] = self._actor.distribution.state_dict()
        parameters_dict['critic_architecture_state_dict'] = self._critic.architecture.state_dict()

        torch.save(parameters_dict, save_path)

    def load_parameters(self, load_path):
        parameters_dict = torch.load(load_path)

        self._actor.architecture.load_state_dict(parameters_dict['actor_architecture_state_dict'])
        self._actor.distribution.load_state_dict(parameters_dict['actor_distribution_state_dict'])
        self._critic.architecture.load_state_dict(parameters_dict['critic_architecture_state_dict'])

        return parameters_dict

    def get_info(self):
        pass

    @property
    def action_mean(self):
        return self._actor.action_mean

    @property
    def shuffle_batch(self):
        return self._shuffle_batch

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic

    @property
    def distribution(self):
        return self.actor.distribution

    @property
    def obs_shape(self):
        return self._actor.obs_shape

    @property
    def action_shape(self):
        return self._actor.action_shape

    @property
    def predict_values_during_act(self):
        return self._predict_values_during_act
