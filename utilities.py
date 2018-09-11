import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import copy
import numpy as np

from model_integrator import model_integration


def normal_np(act, mu, sigma_sq):
    a = np.exp(-(act - mu)**2 / (2. * sigma_sq**2))
    b = 1. / np.sqrt((2. * sigma_sq**2 * np.pi))
    return a * b


def normal_torch(act, mu, sigma_sq):
    a = (-1 * (Variable(act) - mu).pow(2) / (2 * sigma_sq**2)).exp()
    b = 1 / np.sqrt((2 * sigma_sq**2 * np.pi))
    return a * b


# NOTE: not sure if this works for vectorial controls, check
# NOTE: should return only one prob
def select_action(control_mean, control_sigma, train=True):
    '''
    In the constinous space, this means adding a random perturbation to our control
    np.random.normal: Draw random samples from a normal (Gaussian) distribution.
    '''
    if train:  # want control only or also probabilities
        eps = torch.FloatTensor([torch.randn(control_mean.size())])
        control_choice = (control_mean + np.sqrt(control_sigma) * Variable(eps)).data
        prob = normal_torch(control_choice, control_mean, control_sigma)
        log_prob = prob.log()
        # entropy is to explore low likelihood places
        entropy = -0.5 * ((control_sigma + 2 * np.pi).log() + 1)
        return control_choice, log_prob, entropy
    elif not train:
        return control_mean


def pretraining(policy, fixed_actions, params, runs, pert_size,
                timesteps, ti, tf, dtime, initial_state,
                epochs=100):
    '''Trains parametric policy model to resemble desired starting function.'''

    # lists to be filled
    state_range = [[(None, None, None) for step in range(timesteps)] for run in range(runs)]
    control_range = [[None for step in range(timesteps)] for run in range(runs)]

    for run in range(runs):
        t = ti  # define initial time at each episode
        for step in range(timesteps):
            action = fixed_actions[step]  # * (1 + np.random.uniform(-pert_size, pert_size))
            control = {'U': np.float64(action)}

            final_state = model_integration(params, initial_state, control, dtime)
            initial_state = copy.deepcopy(final_state)

            state_range[run][step] = (*final_state, tf-t) # add current time to state
            control_range[run][step] = np.float64(action)

            t = t + dtime  # calculate next time

    # setting data for training
    # passing data as torch vectors
    tensor_states = [Variable(torch.Tensor(state)) for state in state_range]
    tensor_controls = [Variable(torch.Tensor(control)) for control in control_range]

    # training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    # optimizer = torch.optim.LBFGS(policy.parameters(), history_size=10000)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0
        for kk in range(len(tensor_states)):
            for state, control in zip(tensor_states[kk], tensor_controls[kk]):
                output = policy(state)
                loss += criterion(output, control)
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
        loss.backward()
        optimizer.step()

    y1_, y2_, t_ = state_range[0]
    U_ = control_range[0]
    return y1_, y2_, t_, U_


def compute_run(policy_CR, initial_state_CR, params, log_probs_l,
                dtime, timesteps_CR, ti, tf, std_sqr, epi_n,
                plot_CR=False):
    '''Compute a single run given a policy.'''

    if plot_CR:
        U_CR = [None for i in range(timesteps_CR)]
        y1_CR = [None for i in range(timesteps_CR)]
        y2_CR = [None for i in range(timesteps_CR)]
        t_CR = [0 for i in range(timesteps_CR)]

    # define initial conditions
    tj = np.array([ti])
    initial_state = initial_state_CR  # define initial state for Integrator
    # define initial state for Plicy calculation
    initial_state_P = np.hstack([initial_state, tf - tj])
    initial_state_P = Variable(torch.Tensor(initial_state_P))  # make it a torch variable

    for step in range(timesteps_CR):
        controls = policy_CR(initial_state_P)
        if plot_CR:
            action = select_action(controls[0], std_sqr, train=False)
        elif not plot_CR:
            action, log_prob_a, entropy = select_action(controls[0], std_sqr, train=True)
        control = {'U': np.float64(action)}
        final_state = model_integration(
            params, initial_state, control, dtime)
        if not plot_CR:
            log_probs_l[epi_n][step] = log_prob_a  # global var
        initial_state = copy.deepcopy(final_state)
        tj = tj + dtime  # calculate next time
        initial_state_P = np.hstack([initial_state, tf - tj])
        initial_state_P = Variable(torch.Tensor(initial_state_P)
                                   )  # make it a torch variable

        if plot_CR:
            y1_CR[step] = final_state[0]
            y2_CR[step] = final_state[1]
            t_CR[step] += tj
            U_CR[step] = np.float64(action)
    reward_CR = final_state[1]
    if plot_CR:
        return reward_CR, y1_CR, y2_CR, t_CR, U_CR
    if not plot_CR:
        return reward_CR
