from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexNeuralNetwork(nn.Module):
    """Class to generate dynamically a neural network of the given size and dimensions."""

    def __init__(self, layers_sizes):

        super().__init__()

        self.layers_sizes = layers_sizes

        # collect al hidden layers with the specified dimension
        self.hidden = nn.ModuleList()
        for index in range(len(layers_sizes) - 2):
            self.hidden.append(
                nn.Linear(
                    layers_sizes[index], layers_sizes[index + 1], bias=True
                )
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

def neural_policy(states_dim, actions_dim, layers_size, num_layers):
    """Forge neural network where all hidden layers have the same size."""

    layers_sizes = [layers_size for _ in range(num_layers)]
    layers_sizes.insert(0, states_dim)
    layers_sizes.append(actions_dim)

    return FlexNeuralNetwork(layers_sizes)

def extend_policy(policy, num_new_outputs):

    policy.new_out_means = nn.Linear(policy.layers_sizes[-2], num_new_outputs)
    policy.new_out_sigmas = nn.Linear(policy.layers_sizes[-2], num_new_outputs)

    def forward(self, inputs):

        x = inputs
        for layer in self.hidden:
            x = F.relu(layer(x))
        
        means = 5 * torch.sigmoid(self.out_means(x))
        sigmas = 2.5 * torch.sigmoid(self.out_sigmas(x))

        new_means = 5 * torch.sigmoid(self.new_out_means(x))
        new_sigmas = 2.5 * torch.sigmoid(self.new_out_sigmas(x))

        all_means = torch.cat(means, new_means)
        all_sigmas = torch.cat(sigmas, new_sigmas)

        return all_means, all_sigmas
    
    policy.forward = MethodType(forward, policy)

    return policy        
