from os.path import join

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from policies import NeuralNetwork, LinearRegression
from utilities import run_episode, pretraining, training
from plots import plot_state_policy_evol

torch.manual_seed(666)

# specifications for dynamical system
ti = 0
tf = 1
divisions = 20
subinterval = (tf-ti)/divisions
time_points = [ti + div * subinterval for div in range(divisions)]

model_specs = {
    'initial_state':    (1, 0),
    'a':                0.5,
    'b':                1.0,
    'ti':               ti,
    'tf':               tf,
    'divisions':        divisions,
    'subinterval':      subinterval,
    'time_points':      time_points
}

# define policy network and other learning/reporting parameters
hidden_layers_size = 15
policy = NeuralNetwork(hidden_layers_size)
# policy = LinearRegression()

# pretrain policy with linear increasing means and constant standard deviation
pretraining_objective = [div * 5.0 / divisions for div in range(divisions)]
desired_deviation = 2.0

pretraining(
    policy, pretraining_objective, desired_deviation, model_specs,
    learning_rate=1e-1, epochs=100
    )

y1_list, y2_list, U_list = run_episode(policy, model_specs, track_evolution=True)
plot_state_policy_evol(
    time_points, y1_list, y2_list, U_list, objective=pretraining_objective
    )

# ---------------------------------------------------
#                  REINFORCE training
# ---------------------------------------------------

epochs = 100
epoch_episodes = 800

optimizer = optim.Adam(policy.parameters(), lr=5e-2)

epoch_rewards = training(policy, optimizer, epochs, epoch_episodes, model_specs)

plt.plot(epoch_rewards)
plt.ylabel('reward value')
plt.show()
