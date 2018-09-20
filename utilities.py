import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

import copy
import numpy as np

from integrator import model_integration


def normal_np(act, mu, sigma_sq):
    a = np.exp(-(act - mu)**2 / (2. * sigma_sq**2))
    b = 1. / np.sqrt((2. * sigma_sq**2 * np.pi))
    return a * b


def normal_torch(act, mu, sigma_sq):
    a = (-1 * (Tensor(act) - mu).pow(2) / (2 * sigma_sq**2)).exp()
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
        eps = torch.randn(1)
        control_choice = (control_mean + np.sqrt(control_sigma) * eps).data
        prob = normal_torch(control_choice, control_mean, control_sigma)
        log_prob = prob.log()
        # entropy is to explore low likelihood places
        entropy = -0.5 * (np.log(control_sigma +  2 * np.pi) + 1)
        return control_choice, log_prob, entropy
    elif not train:
        return control_mean


def pretraining(policy, fixed_actions, params, runs, pert_size,
                timesteps, ti, tf, dtime, initial_state, learning_rate, epochs):
    '''Trains parametric policy model to resemble desired starting function.'''

    # lists to be filled
    state_runs =   [[None for step in range(timesteps)] for run in range(runs)]
    control_runs = [[None for step in range(timesteps)] for run in range(runs)]

    for run in range(runs):
        t = ti  # define initial time at each episode
        integrated_state = copy.deepcopy(initial_state)
        for step in range(timesteps):
            action = fixed_actions[step] * (1 + np.random.uniform(-pert_size, pert_size))
            control = {'U': np.float64(action)}

            y1, y2 = model_integration(params, integrated_state, control, dtime)

            time_left = tf - t
            state = (y1, y2, time_left) # add current time to state
            state_runs[run][step] = Tensor(state)
            control_runs[run][step] = Tensor([action])

            t = t + dtime  # calculate next time

    # training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # optimizer = torch.optim.LBFGS(policy.parameters(), history_size=10000)

    for epoch in range(epochs): # TODO: merge this loop with first one.
        for run in range(runs):
            optimizer.zero_grad()
            policy_choices = []
            for state in state_runs[run]:
                choice = policy(state)
                policy_choices.append(choice)
            loss = criterion(
                torch.stack(policy_choices).squeeze(),
                torch.stack(control_runs[run]).squeeze()
                )
            loss.backward()
            optimizer.step()

        print("epoch:", epoch, "loss:", loss.item())

    return state_runs, control_runs

def compute_run(policy, initial_state, params, log_probs,
                dtime, timesteps, ti, tf, std_sqr, epi_n,
                plot=False):
    '''Compute a single run given a policy.'''

    if plot:
        U_CR = [None for i in range(timesteps)]
        y1_CR = [None for i in range(timesteps)]
        y2_CR = [None for i in range(timesteps)]
        t_CR = [0 for i in range(timesteps)]

    # define initial conditions
    t = ti
    # define initial state for Policy calculation
    initial_state_P = np.hstack([initial_state, tf - t])
    initial_state_P = Tensor(initial_state_P)

    for step in range(timesteps):
        controls = policy(initial_state_P)
        if plot:
            action = select_action(controls, std_sqr, train=False)
        else:
            action, log_prob_a, entropy = select_action(controls[0], std_sqr, train=True)
        control = {'U': np.float64(action)}
        final_state = model_integration(params, initial_state, control, dtime)
        if not plot:
            log_probs[epi_n][step] = log_prob_a  # global var
        initial_state = copy.deepcopy(final_state)
        t = t + dtime  # calculate next time
        initial_state_P = np.hstack([initial_state, tf - t])
        initial_state_P = Tensor(initial_state_P)

        if plot:
            y1_CR[step] = final_state[0]
            y2_CR[step] = final_state[1]
            t_CR[step] += t
            U_CR[step] = np.float64(action)
    reward_CR = final_state[1]
    if plot:
        return reward_CR, y1_CR, y2_CR, t_CR, U_CR
    else:
        return reward_CR
