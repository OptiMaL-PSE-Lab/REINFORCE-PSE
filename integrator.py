import numpy as np
import scipy.integrate as scp


def state_model(t, state, f_args):

    U, a, b = f_args
    y1, y2 = state

    dev_y1 = -(U + U**2 * a) * y1
    dev_y2 = U * y1 * b

    return [dev_y1, dev_y2]


def model_integration(params, initial_state, controls, time_interval):
    """
    params: dictionary of parameters passed to a model
    initial_state: numpy array of initial state
    controls: numpy array of control actions for this time step
    time_interval: duration of constant control time interval
    """

    # time_interval es el tiempo que se mantiene el control constante
    U = controls['U']
    a = params['a']
    b = params['b']

    ode = scp.ode(state_model)
    ode.set_integrator('dopri5')
    ode.set_f_params([U, a, b])
    ode.set_initial_value(initial_state)  # initial time is zero by default
    final_state = ode.integrate(ode.t + time_interval)
    return final_state
