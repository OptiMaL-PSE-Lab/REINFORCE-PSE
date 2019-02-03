from itertools import chain

import numpy as np
import scipy.integrate as scp


# base class for what we we expect from any ODE model
class ODEModel(object):
    def __init__(self, parameters, controls_dims, state_dims, integrator="dopri5"):
        self.parameters = parameters
        self.controls_dims = controls_dims
        self.state_dims = state_dims
        self.ode = scp.ode(self.system).set_integrator(integrator)

    def _check_dims(self, controls, state):
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

    # f_args should be a concatenation of parameters and controls (parameters first always!)
    @staticmethod  # static required to pass it as argument to scipy ode integrator
    def system(t, state, f_args):
        raise NotImplementedError(
            "The function defining the dynamical system modeled must be specified!"
        )

    def integrate(self, controls, initial_state, integration_time, initial_time=0.0):
        self._check_dims(controls, initial_state)
        f_params = list(chain(self.parameters, controls))  # concatenation
        self.ode.set_f_params(f_params)
        self.ode.set_initial_value(initial_state, initial_time)
        integrated_state = self.ode.integrate(self.ode.t + integration_time)
        return integrated_state


class SimpleModel(ODEModel):
    def __init__(self, parameters):
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
    def __init__(self, parameters):
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


# def simple_system(t, state, f_args):

#     U, a, b = f_args
#     y1, y2 = state

#     y1_prime = -(U + U**2 * a) * y1
#     y2_prime = U * y1 * b

#     return [y1_prime, y2_prime]

# # parameters['subinterval'] = integration_time
# def model_integration(model, parameters, initial_state, integration_time, initial_time=0.0):
#     """
#     params: dictionary of parameters passed to a model
#     initial_state: numpy array of initial state
#     """

#     U = parameters['U']
#     a = parameters['a']
#     b = parameters['b']
#     integration_time = parameters['subinterval']

#     ode = scp.ode(model)
#     ode.set_integrator('dopri5')
#     ode.set_f_params(parameters)
#     ode.set_initial_value(initial_state, initial_time)
#     integrated_state = ode.integrate(ode.t + integration_time)
#     return integrated_state
