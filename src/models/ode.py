from itertools import chain

import numpy as np
import scipy.integrate as scp

from . import AbstractModel

class ODEModel(AbstractModel):
    """Basic class that contains what is expected to be implemented from any ODE model."""

    def __init__(self, states_dims, controls_dims, parameters, integrator="lsoda"):
        self.ode = scp.ode(self.system)
        self.ode.set_integrator(integrator)
        super().__init__(states_dims, controls_dims, parameters)


    @staticmethod  # static required to pass it as argument to scipy ode integrator
    def system(t, state, f_args):
        """
        Vectorial function representing an ODE system.

        derivative array = system(current state)

        f_args is a concatenation of parameters (first always) and controls.
        """
        raise NotImplementedError(
            "The function defining the dynamical system modeled must be specified!"
        )

    def step(self, state, controls, integration_time, initial_time=0.0, runtime_check=True):
        """General scipy integration routine."""
        if runtime_check:
            self._check_dims(state, controls)
        f_params = list(chain(self.parameters, controls))  # concatenation
        self.ode.set_f_params(f_params)
        self.ode.set_initial_value(state, initial_time)
        integrated_state = self.ode.integrate(self.ode.t + integration_time)
        return integrated_state


class SimpleModel(ODEModel):
    def __init__(self, parameters=(0.5, 1.0, 1.0, 1.0)):
        controls_dims = 2
        states_dims = 2
        super().__init__(states_dims, controls_dims, parameters)

    @staticmethod
    def system(t, state, f_args):

        a, b, c, d, U1, U2 = f_args
        y1, _ = state

        y1_prime = -(U1 + a * U1 ** 2) * y1 + d * U2
        y2_prime = (b * U1 - c * U2) * y1

        return [y1_prime, y2_prime]


class ComplexModel(ODEModel):
    def __init__(self, parameters=(0.5, 1.0, 0.7, 0.5)):
        controls_dims = 2
        states_dims = 2
        super().__init__(states_dims, controls_dims, parameters)

    @staticmethod
    def system(t, state, f_args):

        a, b, c, d, U1, U2 = f_args
        y1, y2 = state

        y1_prime = -(U1 + a * U2 ** 2) * y1 + d * U2 * y2 / (y1 + y2)
        y2_prime = (b * U1 - c * U2) * y1

        return [y1_prime, y2_prime]
