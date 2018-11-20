import scipy.integrate as scp


def simple_system(t, state, f_args):

    U, a, b = f_args
    y1, y2 = state

    dev_y1 = -(U + U ** 2 * a) * y1
    dev_y2 = U * y1 * b

    return [y1_prime, y2_prime]

# parameters['subinterval'] = integration_time
def model_integration(model, parameters, initial_state, integration_time, initial_time=0.0):
    """
    params: dictionary of parameters passed to a model
    initial_state: numpy array of initial state
    """

    # time_interval es el tiempo que se mantiene el control constante
    U = parameters["U"]
    a = parameters["a"]
    b = parameters["b"]

    ode = scp.ode(ode_system)
    ode.set_integrator("dopri5")
    ode.set_f_params([U, a, b])
    ode.set_initial_value(initial_state)  # initial time is zero by default
    final_state = ode.integrate(ode.t + parameters["subinterval"])
    return final_state
