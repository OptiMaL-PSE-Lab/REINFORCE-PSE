from math import sqrt

from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 3
ACTION_DIM = 1

# var(X) = 1/12 if X ~ Beta(1,1) ~ U(0,1)
# max var(X) = 1/4 for X ~ Beta with support [0, 1]
# max std(X) = 5/2 for X ~ Beta with support [0, 5]
def mean_std(m, s):
    """Problem specific restrinctions on predicted mean and standard deviation."""
    mean = 5 * sigmoid(m)
    std = 2.5 * sigmoid(s)
    return mean, std


class NeuralNetwork(nn.Module):

    def __init__(self, hidden_dim):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(STATE_DIM, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear3 = nn.Linear(hidden_dim, ACTION_DIM, bias=True)
        self.linear3_ = nn.Linear(hidden_dim, ACTION_DIM, bias=True)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        m, s = self.linear3(x), self.linear3_(x)
        mean, std = mean_std(m, s)
        return mean, std


class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(STATE_DIM, ACTION_DIM)
        self.linear_ = nn.Linear(STATE_DIM, ACTION_DIM)

    def forward(self, x):
        m, s = self.linear(x), self.linear_(x)
        mean, std = mean_std(m, s)
        return mean, std
