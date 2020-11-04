# from torch.quasirandom import SobolEngine
# from gpytorch.models import ExactGP
# import botorch

from . import AbstractModel

# 10 dist in [0,1] --> SobolEngine(dimension=1).draw(10)


class GP_Regression:
    "General Gaussian Process Regression."

    def __init__(self, X, y):
        pass

    def covariance_matrix(self, X_norm, weight, variance_vector):
        pass


class GPModel(AbstractModel):
    """Gaussian Process Approximation of a real model from noisy data."""

    def __init__(self, config):
        pass

    def sample_ode_model(self, ode_model, initial_state, sigma_squared, confi):
        pass

    def from_ode_model(self, ode_model, sigma_squared):
        pass

    def step(self, state, controls, time_step, runtime_check=True):
        pass


# def main():

# if __name__ == "__main__":
#     pass
