"Main execution of whole algorithm."

from copy import deepcopy

from config import set_configuration
from utils import shift_grad_tracking
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


def main():
    "Several runs with different seeds but same random initial conditions (Chebyshev polinomial)."

    coef_ord_combo = random_coeff_order_combinations(2)

    # pretrain policy means based on some random chebyshev polinomial with fixed standar deviation
    identifiers, desired_controls = chebys_tracer(
        coef_ord_combo, CONFIG.time_points, zipped=True
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
    Plotter("SimpleModel", config).plot_everything()
    Plotter("ComplexModel", config).plot_everything()


if __name__ == "__main__":

    main()
