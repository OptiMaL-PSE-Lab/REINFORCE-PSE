from os.path import join

# import ray
import torch
import torch.optim as optim

from integrator import SimpleModel, ComplexModel
from policies import neural_policy, shift_grad_tracking, FlexRNN
from utilities import pretraining, training

# ray.init()  # this needs to be run on main script... not modules
torch.manual_seed(3_141_592)

# -----------------------------------------------------------------------------------------
#                                     MODEL SPECIFICATIONS
# -----------------------------------------------------------------------------------------

# ODE model to use
model = SimpleModel()

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
states_dim = 3
actions_dim = 2
hidden_layers = 2
layers_size = 25

# policy = neural_policy(states_dim, actions_dim, layers_size, hidden_layers)
policy = FlexRNN(states_dim, layers_size, actions_dim)

# -----------------------------------------------------------------------------------------
#                                         PRETRAINING
# -----------------------------------------------------------------------------------------

# pretrain policy with linearly increasing policy means and fixed standar deviation

desired_controls = [
    (div * 5 / divisions, div * 5 / divisions) for div in range(divisions)
]
desired_deviation = 2.0

pretraining(
    model,
    policy,
    desired_controls,
    desired_deviation,
    integration_specs,
    learning_rate=1e-1,
    iterations=150,
)

# -----------------------------------------------------------------------------------------
#                                          TRAINING
# -----------------------------------------------------------------------------------------

opt_specs = {
    "iterations": 200,
    "episode_batch": 100,
    "learning_rate": 5e-3,
    "method": "reinforce",
    "epochs": 1,
}

optimizer = optim.Adam(policy.parameters(), lr=opt_specs["learning_rate"])

training(
    model,
    policy,
    optimizer,
    integration_specs,
    opt_specs,
    record_graphs=True,
    model_id="simple",
)

# -----------------------------------------------------------------------------------------
#                                   MORE COMPLEX MODEL
# -----------------------------------------------------------------------------------------

# freeze all policy layers except last ones
shift_grad_tracking(policy, False)
shift_grad_tracking(policy.out_means, True)
shift_grad_tracking(policy.out_sigmas, True)

# define new parameters
new_model = ComplexModel()
new_optimizer = optim.Adam(policy.parameters(), lr=opt_specs["learning_rate"])
opt_specs.update({"iterations": 100, "learning_rate": 1e-1})

# retrain last layers
training(
    new_model,
    policy,
    new_optimizer,
    integration_specs,
    opt_specs,
    record_graphs=True,
    model_id="complex",
)
