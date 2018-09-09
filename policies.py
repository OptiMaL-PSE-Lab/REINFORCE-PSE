import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    '''
    Policy-network
    Our model will be a feed-forward neural network that takes in the states.
    Outputs mean and std of an action (exploration-explotation).
    '''

    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear3 = nn.Linear(hidden_size, num_outputs, bias=True)

    def forward(self, inputs):
        x = inputs
        x = F.relu(F.dropout(self.linear1(x), p=0.8, training=self.training))
        x = F.relu(F.dropout(self.linear2(x), p=0.8, training=self.training))
        mu = F.relu6(self.linear3(x))
        return mu
