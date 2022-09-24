import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def get_trainable_parameters(self):
        parameters = 0

        for params in list(self.parameters()):
            elements = 1

            for size in list(params.size()):
                elements *= size

            parameters += elements

        return parameters

    @property
    def device(self):
        return next(self.parameters()).device


class MultiLayerPerceptron(BaseNetwork):
    def __init__(self, in_dim, out_dim, hidden_layers, activation=nn.Tanh, dropout=0., init_weights_gain=None,
                 output_bias=True):
        super(MultiLayerPerceptron, self).__init__()

        # Network Parameters
        self._in_dim = in_dim
        self._out_dim = out_dim

        self._layers = [self._in_dim] + hidden_layers + [self._out_dim]

        self._num_layers = len(self._layers)
        self._activation = activation()
        self._dropout = nn.Dropout(p=dropout)

        # Network Layers
        self._fully_connected_layers = nn.ModuleList(
            [nn.Linear(self._layers[layer], self._layers[layer + 1]) for layer in range(self._num_layers - 2)]
        )

        self._fully_connected_layers.append(nn.Linear(self._layers[-2], self._layers[-1], bias=output_bias))

        if init_weights_gain is not None:
            self.init_weights(init_weights_gain)

    def forward(self, t: torch.Tensor):
        for layer in self._fully_connected_layers[:-1]:
            t = self._dropout(self._activation(layer(t)))

        return self._fully_connected_layers[-1](t)

    def init_weights(self, gain):
        [torch.nn.init.orthogonal_(layer.weight, gain=gain) for layer in self._fully_connected_layers]

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def layers(self):
        return self._layers


class GatedRecurrentUnit(BaseNetwork):
    def __init__(self, in_dim, hidden_state_dim, init_weights_gain=None):
        super(GatedRecurrentUnit, self).__init__()

        self._gates_input = nn.Linear(in_dim, hidden_state_dim * 3)
        self._gates_hidden = nn.Linear(hidden_state_dim, hidden_state_dim * 3)

        self._hidden_state_dim = hidden_state_dim

        if init_weights_gain is not None:
            self.init_weights(init_weights_gain)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        r_input, z_input, n_input = self._gates_input(x).chunk(3, 1)
        r_hidden, z_hidden, n_hidden = self._gates_hidden(h).chunk(3, 1)

        r = nn.Sigmoid()(r_input + r_hidden)
        z = nn.Sigmoid()(z_input + z_hidden)
        n = nn.Tanh()(n_input + (r * n_hidden))

        return n + (z * (h - n))

    def init_weights(self, gain):
        for b in range(3):
            torch.nn.init.xavier_uniform_(
                self._gates_input.weight[:, b * self._hidden_state_dim: (b + 1) * self._hidden_state_dim], gain=gain)
            torch.nn.init.orthogonal_(
                self._gates_hidden.weight[:, b * self._hidden_state_dim: (b + 1) * self._hidden_state_dim], gain=gain)


class NetworkArchitecture(BaseNetwork):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError
