from os.path import join

# import ray
import torch
import torch.optim as optim

from integrator import SimpleModel
from policies import NeuralNetwork, LinearRegression
from utilities import pretraining, training, plot_episode

# ray.init()  # this needs to be run on main script... not modules
torch.manual_seed(31_415_926)

# -----------------------------------------------------------------------------------------
#                                     MODEL SPECIFICATIONS
# -----------------------------------------------------------------------------------------

# fixed parameters of the model
parameters = 0.5, 1.0
model = SimpleModel(parameters)

# gather integration details
ti, tf = 0, 1
divisions = 20
subinterval = (tf - ti) / divisions
time_points = [ti + div * subinterval for div in range(divisions)]

integration_specs = {
    "initial_state": (1, 0),
    "ti": ti,
    "tf": tf,
    "divisions": divisions,
    "subinterval": subinterval,
    "time_points": time_points,
}

# define policy network
hidden_layers_size = 25
policy = NeuralNetwork(hidden_layers_size)

# -----------------------------------------------------------------------------------------
#                                         PRETRAINING
# -----------------------------------------------------------------------------------------

# pretrain policy with linearly increasing policy means and fixed standar deviation
pretraining_objective = [div * 5 / divisions for div in range(divisions)]
desired_deviation = 2.0

pretraining(
    model,
    policy,
    pretraining_objective,
    desired_deviation,
    integration_specs,
    learning_rate=1e-1,
    iterations=300,
)

plot_episode(model, policy, integration_specs, objective=pretraining_objective)

# -----------------------------------------------------------------------------------------
#                                          TRAINING
# -----------------------------------------------------------------------------------------

opt_specs = {
    "iterations": 250,
    "episode_batch": 100,
    "learning_rate": 5e-3,
    "method": "reinforce",
    "epochs": 1,
}

optimizer = optim.Adam(policy.parameters(), lr=opt_specs["learning_rate"])

training(model, policy, optimizer, integration_specs, opt_specs, record_graphs=True)
