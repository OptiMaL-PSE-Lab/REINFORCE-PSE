import numpy as np
import scipy.integrate as scp


def model_integration(params, initial_state, controls, time_interval):
    '''
    params: dictionary of parameters passed to a model
    initial_state: numpy array of initial state
    controls: numpy array of control actions for this time step
    time_interval: duration of constant control time interval
    '''
    # time_interval es el tiempo que se mantiene el control constante
    U = controls['U']
    a = params['a']
    b = params['b']

    def state_model(t, initial_state):
        # state vector
        y1 = initial_state[0]
        # y2 = initial_state[1] # not directly involved in ode system

        dev_y1 = -(U + U**2 * a) * y1
        dev_y2 = U * y1 * b
        return np.array([dev_y1, dev_y2], dtype='float64')

    ode = scp.ode(state_model)
    ode.set_integrator('lsoda', nsteps=3000)
    ode.set_initial_value(initial_state, t=0.0)  # initial time always 0
    return ode.integrate(ode.t + time_interval)
