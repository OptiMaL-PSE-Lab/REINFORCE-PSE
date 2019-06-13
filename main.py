from os.path import join

# import ray

from integrator import SimpleModel, ComplexModel
from policies import FlexNN, FlexRNN
from utilities import pretraining, training, shift_grad_tracking

# ray.init()  # this needs to be run on main script... not modules

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

integration_config = {
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
num_layers = 2
layers_size = 25

# policy = FlexNN(states_dim, actions_dim, layers_size, num_layers)
policy = FlexRNN(states_dim, actions_dim, layers_size, num_layers)

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
    integration_config,
    learning_rate=0.1,
    iterations=190,
)

# -----------------------------------------------------------------------------------------
#                                          TRAINING
# -----------------------------------------------------------------------------------------

optim_config = {
    "iterations": 200,
    "episode_batch": 100,
    "learning_rate": 5e-3,
    "method": "reinforce",
    "epochs": 1,
}

training(
    model,
    policy,
    integration_config,
    optim_config,
    record_graphs=True,
    model_id="simple",
)

# -----------------------------------------------------------------------------------------
#                                   MORE COMPLEX MODEL
# -----------------------------------------------------------------------------------------

new_model = ComplexModel()

# freeze all policy layers except last ones
shift_grad_tracking(policy, False)
shift_grad_tracking(policy.out_means, True)
shift_grad_tracking(policy.out_sigmas, True)

# define new parameters
optim_config.update({"iterations": 100, "learning_rate": 1e-2})

# retrain last layers
training(
    new_model,
    policy,
    integration_config,
    optim_config,
    record_graphs=True,
    model_id="complex",
)
