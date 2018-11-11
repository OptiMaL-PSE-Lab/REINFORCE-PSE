from os.path import join
from math import sqrt

# import ray
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from policies import NeuralNetwork, LinearRegression
from utilities import pretraining, training, plot_episode
from plots import plot_reward_evolution

# ray.init()  # this needs to be run on main script... not modules
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

# pretrain policy with uninformative distribution over [0, 5]: X ~ U(0, 5)
pretraining_objective = [div * 5 / divisions for div in range(divisions)]
desired_deviation = 2.5

pretraining(
    policy, pretraining_objective, desired_deviation, model_specs,
    learning_rate=1e-1, iterations=100
    )

plot_episode(policy, model_specs, objective=pretraining_objective)

# ---------------------------------------------------
#                  REINFORCE training
# ---------------------------------------------------

iterations = 200
episode_batch = 100
learning_rate = 1e-3

method = 'ppo'
epochs = 10
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

iteration_rewards = training(
    policy, optimizer, iterations, episode_batch, model_specs,
    method=method, epochs=epochs, record_actions=True
    )

final_plot_path = join('figures', f'reward_method_{method}_iteration{iterations}_batch{episode_batch}_lr{learning_rate}.png')
plot_reward_evolution(
    iteration_rewards, learning_rate, episode_batch,
    store_path=final_plot_path
    )
