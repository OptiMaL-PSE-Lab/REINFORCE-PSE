import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexNN(nn.Module):
    """Class to generate dynamically a neural network."""

    def __init__(self, layers_sizes):

        super().__init__()

        self.layers_sizes = layers_sizes

        # collect al hidden layers with the specified dimension
        self.hidden = nn.ModuleList()
        for index in range(len(layers_sizes) - 2):
            self.hidden.append(
                nn.Linear(layers_sizes[index], layers_sizes[index + 1], bias=True)
            )

        # mean and standar deviation for each output dimension
        self.out_means = nn.Linear(layers_sizes[-2], layers_sizes[-1])
        self.out_sigmas = nn.Linear(layers_sizes[-2], layers_sizes[-1])

    def forward(self, inputs):

        x = inputs
        for layer in self.hidden:
            x = F.relu(layer(x))

        # NOTE: restrictions for Beta distribution

        # var(X) = 1/12 if X ~ Beta(1,1) ~ U(0,1)
        # max var(X) = 1/4 for X ~ Beta with support [0, 1]
        # max std(X) = 5/2 for X ~ Beta with support [0, 5]
        means = 5 * torch.sigmoid(self.out_means(x))
        sigmas = 2.5 * torch.sigmoid(self.out_sigmas(x))

        return means, sigmas


class FlexRNN(nn.Module):
    """Class to generate dynamically a recurrent neural network."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):

        super().__init__()

        # NOTE: for rnn the states should not include time information
        self.input_size = input_size - 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)

        # mean and standar deviation for each output dimension
        self.out_means = nn.Linear(hidden_size, output_size)
        self.out_sigmas = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):

        assert (
            len(inputs) == self.input_size + 1
        ), "states should have time added at the end for convenience, it will be dropped automatically"

        timeless_inputs = inputs[:-1]

        # set initial hidden and cell states
        h0 = torch.zeros(
            self.num_layers, self.input_size, self.hidden_size
        )  # .to(device)
        c0 = torch.zeros(
            self.num_layers, self.input_size, self.hidden_size
        )  # .to(device)

        outputs, _ = self.lstm(timeless_inputs, (h0, c0))  # FIXME: dimension out of range

        # NOTE: restrictions for Beta distribution

        # var(X) = 1/12 if X ~ Beta(1,1) ~ U(0,1)
        # max var(X) = 1/4 for X ~ Beta with support [0, 1]
        # max std(X) = 5/2 for X ~ Beta with support [0, 5]
        means = 5 * torch.sigmoid(self.out_means(outputs[:, -1, :]))
        sigmas = 2.5 * torch.sigmoid(self.out_sigmas(outputs[:, -1, :]))

        return means, sigmas


def neural_policy(states_dim, actions_dim, layers_size, num_layers, topology=None):
    """Forge neural network where all hidden layers have the same size."""

    if topology == "NN":
        layers_sizes = [layers_size for _ in range(num_layers)]
        layers_sizes.insert(0, states_dim)
        layers_sizes.append(actions_dim)
        return FlexNN(layers_sizes)

    elif topology == "RNN":
        return FlexRNN(states_dim, layers_size, num_layers, actions_dim)

    else:
        raise ValueError(f"Supported topologies are 'NN' and 'RNN'.")


def shift_grad_tracking(torch_object, track):
    for param in torch_object.parameters():
        param.requires_grad = track

