import argparse


def set_configuration():
    """
    Read given arguments if any and return corresponding or default configurations
    """

    parser = argparse.ArgumentParser(
        description="All parameters related to policy and optimization."
    )

    parser.add_argument("-pr", "--processes", type=int, default=1)
    parser.add_argument("-div", "--divisions", type=int, default=20)
    parser.add_argument("-nl", "--number-layers", type=int, default=2)
    parser.add_argument("-ls", "--layers-size", type=int, default=25)
    parser.add_argument("-pt", "--policy-type", default="rnn", choices=["nn", "rnn"])
    parser.add_argument("-prelr", "--pre-learning-rate", type=float, default=0.05)
    parser.add_argument("-preit", "--pre-iterations", type=int, default=200)
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-3)
    parser.add_argument("-it", "--iterations", type=int, default=250)
    parser.add_argument("-eb", "--episode-batch", type=int, default=100)
    parser.add_argument(
        "-pgm",
        "--policy-gradient-method",
        choices=["ppo", "reinforce"],
        default="reinforce",
    )
    parser.add_argument("-cs", "--chained-steps", type=int, default=1)
    parser.add_argument("-sg", "--discard-graphics", action="store_true")
    parser.add_argument("-poslr", "--post-learning-rate", type=float, default=1e-2)
    parser.add_argument("-posit", "--post-iterations", type=int, default=100)

    config = parser.parse_args()

    # Add custom attributes to same config object for simplicity
    config.ti = 0
    config.tf = 1
    config.subinterval = (config.tf - config.ti) / config.divisions
    config.time_points = [
        config.ti + div * config.subinterval for div in range(config.divisions)
    ]

    config.initial_state = (1, 0)

    return config
