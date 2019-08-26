"Main execution of whole algorithm."

from copy import deepcopy
from datetime import datetime
import multiprocessing as mp

from config import set_configuration
from utils import grouper, shift_grad_tracking
from initial_controls import random_coeff_order_combinations, chebys_tracer, multilabel_cheby_identifiers
from models.ode import SimpleModel, ComplexModel
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
            model.states_dims, model.controls_dims, config.layers_size, config.number_layers
        )
    elif config.policy_type == "nn":
        policy = FlexNN(
            model.states_dims, model.controls_dims, config.layers_size, config.number_layers
        )

    # pretrain policy means based on some random chebyshev polinomial with fixed standar deviation
    identifiers, desired_controls = chebys_tracer(
        coef_ord_tuple_pair, config.time_points, zipped=True
    )
    desired_deviation = 2.0

    # add initial controls identifiers to config
    config.initial_controls_ids = identifiers

    pretrainer(model, policy, desired_controls, desired_deviation, config)

    chebyshev_labels = multilabel_cheby_identifiers(identifiers)
    data_filepath = (
        f"policy_{policy.__class__.__name__}_"
        f"initial_controls_{chebyshev_labels}_"
        f"datetime_{datetime.now()}"
        ".hdf5"
    )

    trainer(model, policy, config, data_filepath)

    new_model = ComplexModel()

    # freeze all policy layers except last ones
    shift_grad_tracking(policy, False)
    shift_grad_tracking(policy.out_means, True)
    shift_grad_tracking(policy.out_sigmas, True)

    # define new parameters
    config.iterations = config.post_iterations
    config.learning_rate = config.post_learning_rate

    # retrain last layers
    trainer(new_model, policy, config, data_filepath)

    return coef_ord_tuple_pair


def main():

    print(f"Using {CONFIG.processes} processes from {mp.cpu_count()} available.")

    if CONFIG.processes > 1:

        # required to use vscode debugger with "subProcess": true in launch.json configuration
        # https://github.com/microsoft/ptvsd/issues/57#issuecomment-444198292
        # not the nicest method though...
        # https://github.com/microsoft/ptvsd/issues/943#issuecomment-481148979
        # mp.set_start_method("spawn")

        with mp.Pool(processes=CONFIG.processes) as pool:  # uses all available processes by default

            coef_ord_combos = random_coeff_order_combinations(2 * pool._processes)

            for res in pool.imap_unordered(full_process, grouper(coef_ord_combos, 2)):

                print(res)
    
    else:
        coef_ord_combos = random_coeff_order_combinations(2)
        full_process(coef_ord_combos)
    

if __name__ == "__main__":

    main()
