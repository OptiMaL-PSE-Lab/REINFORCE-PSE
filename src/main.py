"Main execution of whole algorithm."

from copy import deepcopy

from config import set_configuration
from utils import shift_grad_tracking
from initial_controls import random_chebys
from integrator import SimpleModel, ComplexModel
from policies import FlexNN, FlexRNN
from training import pretrainer, trainer

# -----------------------------------------------------------------------------------------
#                                     MODEL SPECIFICATIONS
# -----------------------------------------------------------------------------------------

CONFIG = set_configuration()


def main():

    config = deepcopy(CONFIG)

    # ODE model to use
    model = SimpleModel()

    # define policy network
    states_dim = 3
    actions_dim = 2

    if config.policy_type == "rnn":
        policy = FlexRNN(
            states_dim, actions_dim, config.layers_size, config.number_layers
        )
    elif config.policy_type == "nn":
        policy = FlexNN(
            states_dim, actions_dim, config.layers_size, config.number_layers
        )

    # pretrain policy means based on some random chebyshev polinomial with fixed standar deviation
    identifiers, desired_controls = random_chebys(2, config.time_points, zipped=True)
    desired_deviation = 2.0

    # add initial controls identifiers to config
    config.initial_controls_ids = identifiers

    pretrainer(model, policy, desired_controls, desired_deviation, config)

    trainer(model, policy, config)

    new_model = ComplexModel()

    # freeze all policy layers except last ones
    shift_grad_tracking(policy, False)
    shift_grad_tracking(policy.out_means, True)
    shift_grad_tracking(policy.out_sigmas, True)

    # define new parameters
    config.iterations = config.post_iterations
    config.learning_rate = config.post_learning_rate

    # retrain last layers
    trainer(new_model, policy, config)


if __name__ == "__main__":

    main()
