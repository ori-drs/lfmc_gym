import numpy as np
import torch.nn

import raisim_gym_torch.algo.ppo.networks as networks


class ActorNetworkArchitecture(networks.NetworkArchitecture):
    def __init__(self, input_dim, hidden_layers, output_dim, activation, network_weights_gain=np.sqrt(2)):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

        if hidden_layers is not None:
            self._network = networks.MultiLayerPerceptron(
                in_dim=input_dim, hidden_layers=hidden_layers, out_dim=output_dim,
                activation=activation, dropout=0., init_weights_gain=network_weights_gain)
        else:
            self._network = torch.nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)
            torch.nn.init.orthogonal_(self._network.weight, gain=network_weights_gain)

    def forward(self, t):
        return self._network.forward(t)

    @property
    def input_shape(self):
        return [self._input_dim]

    @property
    def output_shape(self):
        return [self._output_dim]


class CriticNetworkArchitecture(networks.NetworkArchitecture):
    def __init__(self, input_dim, hidden_layers, activation, network_weights_gain=np.sqrt(2)):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = 1

        if hidden_layers is not None:
            self._network = networks.MultiLayerPerceptron(
                in_dim=input_dim, hidden_layers=hidden_layers, out_dim=1,
                activation=activation, dropout=0., init_weights_gain=network_weights_gain)
        else:
            self._network = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)
            torch.nn.init.orthogonal_(self._network.weight, gain=network_weights_gain)

    def forward(self, t):
        return self._network.forward(t)

    @property
    def input_shape(self):
        return [self._input_dim]

    @property
    def output_shape(self):
        return [self._output_dim]
