
from config import set_configuration
from utils import shift_grad_tracking
from static_controls import random_chebys
from integrator import SimpleModel, ComplexModel
from policies import FlexNN, FlexRNN
from training import pretrainer, trainer

# -----------------------------------------------------------------------------------------
#                                     MODEL SPECIFICATIONS
# -----------------------------------------------------------------------------------------

config = set_configuration()

# ODE model to use
model = SimpleModel()

# define policy network
states_dim = 3
actions_dim = 2

if config.policy_type == "rnn":
    policy = FlexRNN(states_dim, actions_dim, config.layers_size, config.number_layers)
elif config.policy_type == "nn":
    policy = FlexNN(states_dim, actions_dim, config.layers_size, config.number_layers)


# -----------------------------------------------------------------------------------------
#                                         PRETRAINING
# -----------------------------------------------------------------------------------------

# pretrain policy with linearly increasing policy means and fixed standar deviation

desired_controls = random_chebys(2, config.time_points, zipped=True)
desired_deviation = 2.0

pretrainer(
    model,
    policy,
    desired_controls,
    desired_deviation,
    config,
)

# -----------------------------------------------------------------------------------------
#                                          TRAINING
# -----------------------------------------------------------------------------------------

trainer(model,policy,config)

# -----------------------------------------------------------------------------------------
#                                   COMPLEX MODEL
# -----------------------------------------------------------------------------------------

new_model = ComplexModel()

# freeze all policy layers except last ones
shift_grad_tracking(policy, False)
shift_grad_tracking(policy.out_means, True)
shift_grad_tracking(policy.out_sigmas, True)

# define new parameters
config.iterations = config.post_iterations
config.learning_rate = config.post_learning_rate

# retrain last layers
trainer(new_model,policy,config)
