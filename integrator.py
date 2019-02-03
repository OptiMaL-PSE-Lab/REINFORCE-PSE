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
        # TODO: allow non-iterabe controls and states
        assert self.controls_dims == len(controls), f"This model expects {self.controls_dims} controls!"
        assert self.state_dims == len(state), f"This model expects {self.state_dims} controls!"

    @staticmethod
    def system(t, state, f_args):  # f_args should be a concatenation of parameters and controls (parameters first always!)
        raise NotImplementedError("The function defining the dynamical system modeled must be specified!")

    def integrate(self, controls, initial_state, integration_time, initial_time=0.0):
        self._check_dims(controls, initial_state)
        f_params = list(chain(self.parameters, controls))  # concatenation
        self.ode.set_f_params(f_params)
        self.ode.set_initial_value(initial_state, initial_time)
        integrated_state = self.ode.integrate(self.ode.t + integration_time)
        return integrated_state


class SimpleModel(ODEModel):

    def __init__(self, parameters, controls_dims, state_dims):
        super().__init__(parameters, controls_dims, state_dims)

    @staticmethod
    def system(t, state, f_args):
        a, b, U = f_args
        y1, _ = state

        y1_prime = -(U + U**2 * a) * y1
        y2_prime = U * y1 * b

        return [y1_prime, y2_prime]

S = SimpleModel([0.5,1.0], 1, 2)
S.integrate([5.0], (1,0), 0.5)

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
