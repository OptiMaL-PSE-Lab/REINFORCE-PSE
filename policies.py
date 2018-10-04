from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 3
OUTPUT_DIM = 2  # if changed, adjust end of forward method in policies


def mean_std(m, s):
    """Problem specific restrinctions on predicted mean and standard deviation."""
    mean = 5 * sigmoid(m)
    std = 25/12 * sigmoid(s)
    return mean, std


class NeuralNetwork(nn.Module):

    def __init__(self, hidden_dim):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(INPUT_DIM, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear3 = nn.Linear(hidden_dim, OUTPUT_DIM, bias=True)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        m, s = self.linear3(x)
        mean, std = mean_std(m, s)
        return mean, std


class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(INPUT_DIM, OUTPUT_DIM)

    def forward(self, x):
        m, s = self.linear(x)
        mean, std = mean_std(m, s)
        return mean, std
