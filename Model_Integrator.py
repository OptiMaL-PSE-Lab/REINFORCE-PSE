import numpy as np
import scipy.integrate as scp

def model_integration(params, initial_state, controls, dtime):
    '''
    params: dictionary of parameters passed to a model
    initial_state: numpy array of initial state
    controls: numpy array of control actions for this time step
    dtime: duration of time step
    '''
    # dtime es el tiempo que se mantiene el control constante
    U_u = controls['U_u']
    a_p = params['a_p']
    b_p=params['b_p']

    def ode_system(t, initial_state):
        #state vector
        y1_s = initial_state[0]
        # y2_s = initial_state[1] # not directly involved in ode system

        dev_y1 = -(U_u+U_u**2*a_p)*y1_s
        dev_y2 = U_u*y1_s*b_p
        return np.array([dev_y1, dev_y2], dtype='float64')

    ode = scp.ode(ode_system)
    ode.set_integrator('lsoda', nsteps=3000)
    ode.set_initial_value(initial_state, t=0.0) # initial time always 0
    return ode.integrate(ode.t + dtime)
