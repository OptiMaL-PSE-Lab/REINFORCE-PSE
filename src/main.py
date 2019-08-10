"Main execution of whole algorithm."

from copy import deepcopy
from multiprocessing import Pool

from config import set_configuration
from utils import grouper, shift_grad_tracking
from initial_controls import random_coeff_order_combinations, chebys_tracer
from integrator import SimpleModel, ComplexModel
from policies import FlexNN, FlexRNN
from training import pretrainer, trainer

# -----------------------------------------------------------------------------------------
#                                     MODEL SPECIFICATIONS
# -----------------------------------------------------------------------------------------


CONFIG = set_configuration()


def full_process(coef_ord_tuple_pair):

    config = deepcopy(CONFIG)

    # ODE model to start with
    model = SimpleModel()

    # create desired policy
    if config.policy_type == "rnn":
        policy = FlexRNN(
            model.state_dims, model.controls_dims, config.layers_size, config.number_layers
        )
    elif config.policy_type == "nn":
        policy = FlexNN(
            model.state_dims, model.controls_dims, config.layers_size, config.number_layers
        )

    # pretrain policy means based on some random chebyshev polinomial with fixed standar deviation
    identifiers, desired_controls = chebys_tracer(
        coef_ord_tuple_pair, config.time_points, zipped=True
    )
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

    return coef_ord_tuple_pair


def main():

    with Pool(processes=CONFIG.processes) as pool:  # uses all available processes by default

        coef_ord_combos = random_coeff_order_combinations(2 * pool._processes)

        for res in pool.imap_unordered(full_process, grouper(coef_ord_combos, 2)):

            print(res)


if __name__ == "__main__":

    main()
