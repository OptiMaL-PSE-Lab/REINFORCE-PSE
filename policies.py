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

    # NOTE: hidden state and additional None return required for common interface with RNN 
    def forward(self, inputs, hidden_state=None):

        x = inputs
        for layer in self.hidden:
            x = F.relu(layer(x))

        # NOTE: restrictions for Beta distribution

        # var(X) = 1/12 if X ~ Beta(1,1) ~ U(0,1)
        # max var(X) = 1/4 for X ~ Beta with support [0, 1]
        # max std(X) = 5/2 for X ~ Beta with support [0, 5]
        means = 5 * torch.sigmoid(self.out_means(x))
        sigmas = 2.5 * torch.sigmoid(self.out_sigmas(x))

        return (means, sigmas), None


class FlexRNN(nn.Module):
    """Class to generate dynamically a recurrent neural network."""

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        # NOTE: for rnn the states should not include time information
        self.input_size = input_size - 1
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(self.input_size, self.hidden_size)

        # mean and standar deviation for each output dimension
        self.out_means = nn.Linear(hidden_size, output_size)
        self.out_sigmas = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden_state=None):

        assert (
            len(inputs) == self.input_size + 1
        ), "states should have time added at the end for convenience, it will be dropped automatically"

        timeless_inputs = inputs[:-1].unsqueeze(0) # reshape for batch size = 1

        if hidden_state is None:
            # set initial hidden states
            h0 = torch.randn(1, self.hidden_size)
            c0 = torch.randn(1, self.hidden_size)
            hidden_state = (h0, c0)

        h1, c1 = self.cell(timeless_inputs, hidden_state)  # FIXME!!!

        # NOTE: restrictions for Beta distribution

        # var(X) = 1/12 if X ~ Beta(1,1) ~ U(0,1)
        # max var(X) = 1/4 for X ~ Beta with support [0, 1]
        # max std(X) = 5/2 for X ~ Beta with support [0, 5]
        means = 5 * torch.sigmoid(self.out_means(h1))
        sigmas = 2.5 * torch.sigmoid(self.out_sigmas(h1))

        return (means, sigmas), (h1, c1)


def neural_policy(states_dim, actions_dim, layers_size, num_layers):
    """Forge neural network where all hidden layers have the same size."""

    layers_sizes = [layers_size for _ in range(num_layers)]
    layers_sizes.insert(0, states_dim)
    layers_sizes.append(actions_dim)
    return FlexNN(layers_sizes)


def shift_grad_tracking(torch_object, track):
    for param in torch_object.parameters():
        param.requires_grad = track

