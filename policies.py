import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):

    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear3 = nn.Linear(hidden_size, num_outputs, bias=True)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = 5 * F.sigmoid(self.linear3(x))
        return mu


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        out = 5 * F.sigmoid(x)
        return out
