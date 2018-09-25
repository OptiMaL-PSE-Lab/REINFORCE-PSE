import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Normal

from integrator import model_integration


# TODO: let it work with any given distribution
def select_action(control_mean, control_sigma):
    """
    In the continuous space, this means adding a random perturbation to our control
    """
    dist = Normal(control_mean, control_sigma)
    control_choice = dist.sample()
    log_prob = dist.log_prob(control_choice)
    entropy = dist.entropy()
    return control_choice, log_prob, entropy


def pretraining(policy, objective_actions, ode_params, initial_state,
                time_divisions, ti, tf, dtime, learning_rate, epochs, pert_size=0.1):
    """Trains parametric policy model to resemble desired starting function."""

    # training parameters
    criterion = nn.MSELoss(reduction='elementwise_mean')
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    objective_tensor = torch.tensor(objective_actions)
    # optimizer = torch.optim.LBFGS(policy.parameters(), history_size=10000)

    # lists to be filled
    states =   [None for step in range(time_divisions)]
    controls = [None for step in range(time_divisions)]

    for epoch in range(epochs):
        t = ti  # define initial time at each episode
        integrated_state = copy.deepcopy(initial_state)
        optimizer.zero_grad()
        for division in range(time_divisions):

            action = np.random.normal(loc=objective_actions[division], scale=pert_size)
            control_dict = {'U': np.float64(action)}

            y1, y2 = model_integration(ode_params, integrated_state, control_dict, dtime)

            time_left = tf - t
            state = Tensor((y1, y2, time_left)) # add time left to state

            states[division] = state
            controls[division] = policy(state)

            t = t + dtime  # calculate next time

        input_controls = torch.stack(controls).squeeze()
        loss = criterion(objective_tensor, input_controls)
        loss.backward()
        optimizer.step()

        print("epoch:", epoch, "\t loss:", loss.item())

    return states, controls


def run_episode(policy, initial_state, ode_params, log_probs, rewards, epi_n,
                dtime, divisions, ti, tf, std_sqr, return_evolution=False):
    """Compute a single run given a policy."""

    if return_evolution:
        y1 = [None for i in range(divisions)]
        y2 = [None for i in range(divisions)]
        U = [None for i in range(divisions)]

    # define initial conditions
    t = ti
    initial_state_P = Tensor((*initial_state, tf - t))

    for step in range(divisions):
        controls = policy(initial_state_P)
        action, log_prob, _ = select_action(controls[0], std_sqr)

        control = {'U': np.float64(action)}
        final_state = model_integration(ode_params, initial_state, control, dtime)

        if not return_evolution:
            log_probs[epi_n][step] = log_prob

        initial_state = copy.deepcopy(final_state)
        t = t + dtime  # calculate next time
        initial_state_P = Tensor((*initial_state, tf - t))

        if return_evolution:
            y1[step] = final_state[0]
            y2[step] = final_state[1]
            U[step] = np.float64(action)

    if return_evolution:
        return y1, y2, U
    else:
        rewards[epi_n] = final_state[1]
        return None
