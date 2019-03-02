from os.path import join

# import ray
import torch
import torch.optim as optim

from integrator import SimpleModel
from policies import neural_policy, extend_policy, shift_grad_tracking
from utilities import pretraining, training, plot_episode

# ray.init()  # this needs to be run on main script... not modules
torch.manual_seed(3_141_926)

# -----------------------------------------------------------------------------------------
#                                     MODEL SPECIFICATIONS
# -----------------------------------------------------------------------------------------

# ODE model to use
model = SimpleModel()

# gather integration details
ti, tf = 0, 1
divisions = 25
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
states_dim = 3
actions_dim = 1
hidden_layers = 2
layers_size = 25
policy = neural_policy(states_dim, actions_dim, layers_size, hidden_layers)

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
    iterations=150,
)

plot_episode(model, policy, integration_specs, objective=pretraining_objective)

# -----------------------------------------------------------------------------------------
#                                          TRAINING
# -----------------------------------------------------------------------------------------

opt_specs = {
    "iterations": 150,
    "episode_batch": 100,
    "learning_rate": 5e-3,
    "method": "reinforce",
    "epochs": 1,
}

optimizer = optim.Adam(policy.parameters(), lr=opt_specs["learning_rate"])

training(model, policy, optimizer, integration_specs, opt_specs, record_graphs=True)

# TODO: freeze old layers while training new ones
new_controls = 1
new_policy = extend_policy(policy, new_controls)

# -----------------------------------------------------------------------------------------
#                                      EXTEND POLICY
# -----------------------------------------------------------------------------------------

# freeze all layers except added ones
shift_grad_tracking(new_policy, False)
shift_grad_tracking(new_policy.new_out_means, True)
shift_grad_tracking(new_policy.new_out_sigmas, True)

new_optimizer = optim.Adam(new_policy.parameters(), lr=opt_specs["learning_rate"])
# training(model, new_policy, new_optimizer, integration_specs, opt_specs, record_graphs=False)
