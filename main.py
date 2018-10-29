from os.path import join
from math import sqrt

# import ray
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from policies import NeuralNetwork, LinearRegression
from utilities import pretraining, training, plot_policy_sample

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
pretraining_objective = [2.5 for div in range(divisions)]
desired_deviation = 2.5 / sqrt(3)

pretraining(
    policy, pretraining_objective, desired_deviation, model_specs,
    learning_rate=1e-1, epochs=100
    )

plot_policy_sample(policy, model_specs, objective=pretraining_objective)

# ---------------------------------------------------
#                  REINFORCE training
# ---------------------------------------------------

epochs = 150
episode_batch = 500
learning_rate = 2e-2

optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

epoch_rewards = training(policy, optimizer, epochs, episode_batch, model_specs)

plt.plot(epoch_rewards)
plt.title(f'batch size:{episode_batch} lr:{learning_rate}')
plt.xlabel('epoch')
plt.ylabel('reward')
plt.savefig(f'figures/reward_epoch{epochs}_batch{episode_batch}_lr{learning_rate}.png')
plt.show()
