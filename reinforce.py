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
policy = NeuralNetwork(hidden_layers_size)
# policy = LinearRegression()

# pretrain policy with linear policy
pretraining_objective = [div * 5.0 / divisions for div in range(divisions)]
initial_state = np.array([1, 0])

pretraining(
    policy, pretraining_objective, ode_params, initial_state, divisions, ti, tf, dtime,
    learning_rate=1e-1, epochs=100, pert_size=0.0
    )

y1_s, y2_s, U_s = run_episode(
    policy, initial_state, ode_params, 
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

optimizer = optim.Adam(policy.parameters(), lr=1e-4)

epoch_rewards = training(
    policy, optimizer, epochs, epoch_episodes,
    ode_params, dtime, divisions, ti, tf
    )

plt.plot(epoch_rewards)
plt.ylabel('reward value')
plt.show()
