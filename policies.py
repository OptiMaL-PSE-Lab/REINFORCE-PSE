import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTE: restrictions for Beta distribution **

# var(X) = 1/12 if X ~ Beta(1,1) ~ U(0,1)
# max var(X) = 1/4 for X ~ Beta with support [0, 1]
# max std(X) = 5/2 for X ~ Beta with support [0, 5]


class FlexNN(nn.Module):
    """Class to generate dynamically a neural network."""

    def __init__(self, states_dim, actions_dim, layers_size, num_layers):

        super().__init__()

        # construct sequence of layers sizes
        layers_sizes = [layers_size for _ in range(num_layers)]
        layers_sizes.insert(0, states_dim)
        layers_sizes.append(actions_dim)
        self.layers_sizes = layers_sizes

        # collect al hidden layers with the specified dimension
        self.hidden = nn.ModuleList()
        for index in range(len(layers_sizes) - 2):
            self.hidden.append(nn.Linear(layers_sizes[index], layers_sizes[index + 1]))

        # mean and standar deviation for each output dimension
        self.out_means = nn.Linear(layers_sizes[-2], layers_sizes[-1])
        self.out_sigmas = nn.Linear(layers_sizes[-2], layers_sizes[-1])

    # hidden state and additional None return required for common interface with RNN
    def forward(self, inputs, hidden_state=None):

        x = inputs
        for layer in self.hidden:
            x = F.relu(layer(x))

        # restrictions for Beta distribution **
        means = 5 * torch.sigmoid(self.out_means(x))
        sigmas = 2.5 * torch.sigmoid(self.out_sigmas(x))

        return (means, sigmas), None


class FlexRNN(nn.Module):
    """Class to generate dynamically a recurrent neural network."""

    def __init__(self, input_size, output_size, hidden_size, num_layers):

        super().__init__()

        # NOTE: for rnn the states should not include time information
        self.input_size = input_size - 1
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cell = nn.LSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layers
        )

        # mean and standar deviation for each output dimension
        self.out_means = nn.Linear(self.hidden_size, self.output_size)
        self.out_sigmas = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden_state=None):

        assert (
            len(inputs) == self.input_size + 1
        ), "states should have time added at the end for convenience, it will be dropped automatically"

        # reshape for seq_len, batch = 1, 1
        timeless_inputs = inputs[:-1].view(1, 1, -1)

        output, hidden_state = self.cell(timeless_inputs, hidden_state)

        # restrictions for Beta distribution **
        means = 5 * torch.sigmoid(self.out_means(output.squeeze()))
        sigmas = 2.5 * torch.sigmoid(self.out_sigmas(output.squeeze()))

        return (means, sigmas), hidden_state


def shift_grad_tracking(torch_object, track):
    for param in torch_object.parameters():
        param.requires_grad = track

