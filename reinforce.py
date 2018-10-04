import os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import Tensor

from policies import NeuralNetwork, LinearRegression
from utilities import pretraining, run_episode, training
from plots import plot_state_policy_evol

np.random.seed(seed=666)
torch.manual_seed(666)

# specifications for dynamic system
ti = 0  # define initial time
tf = 1  # define final time
divisions = 20
dtime = (tf-ti)/divisions
time_array = [ti + div * dtime for div in range(divisions)]

ode_params = {'a': 0.5, 'b': 1}

# define policy network and other learning/reporting parameters
hidden_layers_size = 15
input_size = 3
output_size = 1
policy = NeuralNetwork(hidden_layers_size, input_size, output_size)
# policy = LinearRegression(input_size, output_size)

# pretrain policy with linear policy
pretraining_objective = [div * 5.0 / divisions for div in range(divisions)]
initial_state = np.array([1, 0])

state_range_PT, control_range_PT = pretraining(
    policy, pretraining_objective, ode_params, initial_state, divisions, ti, tf, dtime,
    learning_rate=1e-1, epochs=100, pert_size=0.0
    )

y1_s, y2_s, U_s = run_episode(
    policy, initial_state, ode_params, 0.0001,
    dtime, divisions, ti, tf, track_evolution=True
    )
plot_state_policy_evol(time_array, y1_s, y2_s, U_s, objective=pretraining_objective)

# ---------------------------------------------------
#                  REINFORCE training
# ---------------------------------------------------

# problem parameters
epochs = 500
epoch_episodes = 200

# NOTE: total_episodes = epochs * (epoch_episodes + 1)

sigma = 5/4
sigma_reduction = 0.999

optimizer = optim.Adam(policy.parameters(), lr=1e-1)

epoch_rewards = training(
    policy, optimizer, epochs, epoch_episodes, sigma, sigma_reduction,
    ode_params, dtime, divisions, ti, tf
    )

plt.plot(epoch_rewards)
plt.ylabel('reward value')
plt.show()
