from itertools import chain

import numpy as np
import scipy.integrate as scp


class ODEModel(object):
    """Basic class that contains what is expected to be implemented from any ODE model."""

    def __init__(self, parameters, controls_dims, state_dims, integrator="dopri5"):
        self.parameters = parameters
        self.controls_dims = controls_dims
        self.state_dims = state_dims
        self.ode = scp.ode(self.system)
        self.ode.set_integrator(integrator)

    def _check_dims(self, controls, state):
        """Runtime check of control and state dimensions."""
        try:
            iter(controls)
            iter(state)
        except TypeError:
            raise TypeError(
                "Please use containers for controls and states: value --> [value]."
            )
        else:
            assert self.controls_dims == len(
                controls
            ), f"This model expects {self.controls_dims} controls!"
            assert self.state_dims == len(
                state
            ), f"This model expects {self.state_dims} controls!"

    @staticmethod  # static required to pass it as argument to scipy ode integrator
    def system(t, state, f_args):
        """
        Vectorial function representing an ODE system.
        
        * derivative array = system(current state)
        
        f_args is a concatenation of parameters (first always) and controls.
        """
        raise NotImplementedError(
            "The function defining the dynamical system modeled must be specified!"
        )

    def integrate(self, controls, initial_state, integration_time, initial_time=0.0):
        """General scipy integration routine."""
        self._check_dims(controls, initial_state)
        f_params = list(chain(self.parameters, controls))  # concatenation
        self.ode.set_f_params(f_params)
        self.ode.set_initial_value(initial_state, initial_time)
        integrated_state = self.ode.integrate(self.ode.t + integration_time)
        return integrated_state


class SimpleModel(ODEModel):
    def __init__(self, parameters=(0.5, 1.0)):
        control_dims = 1
        state_dims = 2
        super().__init__(parameters, control_dims, state_dims)

    @staticmethod
    def system(t, state, f_args):

        a, b, U = f_args
        y1, _ = state

        y1_prime = -(U + a * U ** 2) * y1
        y2_prime = b * U * y1

        return [y1_prime, y2_prime]


class ComplexModel(ODEModel):
    def __init__(self, parameters=(0.5, 1.0)):
        control_dims = 2
        state_dims = 2
        super().__init__(parameters, control_dims, state_dims)

    @staticmethod
    def system(t, state, f_args):

        a, b, U1, U2 = f_args
        y1, y2 = state

        y1_prime = -(U1 + a * U2 ** 2) * y1 + 0.1 * U2 ** 2 / (y1 + y2)
        y2_prime = b * U1 * y1 - 0.7 * U2 * y1

        return [y1_prime, y2_prime]
