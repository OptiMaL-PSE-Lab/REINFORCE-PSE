"Main execution of whole algorithm."

from copy import deepcopy
import multiprocessing as mp

from config import set_configuration
from utils import grouper, shift_grad_tracking
from initial_controls import (
    random_coeff_order_combinations,
    chebys_tracer,
    multilabel_cheby_identifiers,
)
from models.ode import SimpleModel, ComplexModel
from training import Trainer
from plots import Plotter

CONFIG = set_configuration()


def training_pipeline(config, desired_controls, desired_deviation):
    """
    Pretrain policy with given control sequence and simple model,
    then train again last layers with complex model.
    """

    # ODE model to start with
    model = SimpleModel()

    trainer = Trainer(model, config)

    # pretrain policy with given control sequence
    trainer.pretrain(desired_controls, desired_deviation)

    # on-policy training
    trainer.train()

    # more complex variation of same ODE model
    new_model = ComplexModel()
    trainer.set_model(new_model)

    # freeze all policy layers except last ones
    shift_grad_tracking(trainer.policy, False)
    shift_grad_tracking(trainer.policy.out_means, True)
    shift_grad_tracking(trainer.policy.out_sigmas, True)

    # retrain on-policy last layers
    trainer.train(post_training=True)


def full_process(coef_ord_tuple_pair):
    "Several runs with different seeds but same initial conditions."

    # pretrain policy means based on some random chebyshev polinomial with fixed standar deviation
    identifiers, desired_controls = chebys_tracer(
        coef_ord_tuple_pair, CONFIG.time_points, zipped=True
    )
    desired_deviation = 2.0

    config = deepcopy(CONFIG)

    # add initial controls identifiers to config
    labels = multilabel_cheby_identifiers(identifiers)
    config.initial_controls_labels = labels
    print(f"Initial controls {labels}")

    # repeat simulation with different seeds
    for _ in range(config.distinct_seeds):
        training_pipeline(config, desired_controls, desired_deviation)

    # plot results
    plotter = Plotter(config)
    plotter.playground()

    return coef_ord_tuple_pair


def main():

    print(f"Using {CONFIG.processes} processes from {mp.cpu_count()} available.")

    # NOTE: maybe generalize this to n-component systems (for now 2 components)
    coef_ord_combos = random_coeff_order_combinations(2 * CONFIG.processes)

    if CONFIG.processes > 1:

        # required to use vscode debugger with "subProcess": true in launch.json configuration
        # https://github.com/microsoft/ptvsd/issues/57#issuecomment-444198292
        # not the nicest method though...
        # https://github.com/microsoft/ptvsd/issues/943#issuecomment-481148979
        # mp.set_start_method("spawn")

        with mp.Pool(
            processes=CONFIG.processes
        ) as pool:  # uses all available processes by default

            for res in pool.imap_unordered(full_process, grouper(coef_ord_combos, 2)):
                print(res)
    else:
        full_process(coef_ord_combos)


if __name__ == "__main__":

    main()
