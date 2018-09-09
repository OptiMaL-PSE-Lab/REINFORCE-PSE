import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import copy
import numpy as np

from model_integrator import model_integration

pi = Variable(torch.FloatTensor([np.pi]))


def normal_np(act, mu, sigma_sq):
    a = np.exp(-(act - mu)**2 / (2. * sigma_sq**2))
    b = 1. / np.sqrt((2. * sigma_sq**2 * np.pi))
    return a * b


def normal_torch(act, mu, sigma_sq):
    a = (-1 * (Variable(act) - mu).pow(2) / (2 * sigma_sq**2)).exp()
    b = 1 / np.sqrt((2 * sigma_sq**2 * pi))
    return a * b


# NOTE: not sure if this works for vectorial controls, check
# NOTE: should return only one prob
def select_action(control_mean, control_sigma, train=True):
    '''
    Select an action accordingly to an epsilon greedy policy.
    In the constinous space, this means adding a random perturbation to our control
    np.random.normal: Draw random samples from a normal (Gaussian) distribution.
    '''
    if train:  # want control only or also probabilities
        eps = torch.FloatTensor([torch.randn(control_mean.size())])
        control_choice = (control_mean + np.sqrt(control_sigma) * Variable(eps)).data
        prob = normal_torch(control_choice, control_mean, control_sigma)
        log_prob = prob.log()
        # entropy is to explore low likelihood places
        entropy = -0.5 * ((control_sigma + 2 * pi).log() + 1)
        return control_choice, log_prob, entropy
    elif not train:
        return control_mean


def pretraining(policy_PT, inputs, params, runs_PT, pert_size,
                t_steps, ti, tf, dtime,
                epoch_n=100, initial_state_I=np.array([1, 0])):
    '''Trains parametric policy model to resemble desired starting function.'''

    # lists to be filled
    y1_PT = [[None for i_PT in range(t_steps)] for i_PT in range(runs_PT)]
    y2_PT = [[None for i_PT in range(t_steps)] for i_PT in range(runs_PT)]
    t_PT = [[None for i_PT in range(t_steps)] for i_PT in range(runs_PT)]
    U_u_PT = [[None for i_PT in range(t_steps)] for i_PT in range(runs_PT)]

    for i_episode in range(runs_PT):
        tj = np.array([ti])  # define initial time at each episode
        for step_j in range(t_steps):
            controls = inputs[step_j]  # * (1 + np.random.uniform(-pert_size, pert_size))
            action = controls
            control = {'U_u': np.float64(action)}
            final_state = model_integration(
                params, initial_state_I, control, dtime
                )
            initial_state_I = copy.deepcopy(final_state)
            tj = tj + dtime  # calculate next time
            y1_PT[i_episode][step_j] = final_state[0]
            y2_PT[i_episode][step_j] = final_state[1]
            t_PT[i_episode][step_j] = tf - tj
            U_u_PT[i_episode][step_j] = np.float64(action)
    # setting data for training
    y_data = [[(U_u_PT[j][i]) for i in range(len(U_u_PT[j]))]
              for j in range(len(U_u_PT))]
    x_data = [[(y1_PT[j][i], y2_PT[j][i], t_PT[j][i])
               for i in range(len(y1_PT[j]))]
              for j in range(len(y1_PT))]
    # passing data as torch vectors
    inputs_l = [Variable(torch.Tensor(x_data[i])) for i in range(len(x_data))]
    labels_l = [Variable(torch.Tensor(y_data[j])) for j in range(len(y_data))]
    # training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy_PT.parameters(), lr=1e-2)
    # optimizer = torch.optim.LBFGS(policy_PT.parameters(), history_size=10000)

    for PT_epoch in range(epoch_n):
        optimizer.zero_grad()
        PT_loss = 0
        for kk in range(len(inputs_l)):
            for inpt, label in zip(inputs_l[kk], labels_l[kk]):
                output = policy_PT(inpt)
                PT_loss += criterion(output, label)
        print("epoch: %d, loss: %1.3f" % (PT_epoch + 1, PT_loss.data[0]))
        PT_loss.backward()
        optimizer.step()
    return y1_PT[0], y2_PT[0], t_PT[0], U_u_PT[0]


def compute_run(policy_CR, initial_state_CR, params, log_probs_l,
                dtime, t_steps_CR, ti, tf, std_sqr, epi_n,
                plot_CR=False):
    '''Compute a single run given a policy.'''

    if plot_CR:
        U_u_CR = [None for i in range(t_steps_CR)]
        y1_CR = [None for i in range(t_steps_CR)]
        y2_CR = [None for i in range(t_steps_CR)]
        t_CR = [0 for i in range(t_steps_CR)]

    # define initial conditions
    tj = np.array([ti])
    initial_state_I = initial_state_CR  # define initial state for Integrator
    # define initial state for Plicy calculation
    initial_state_P = np.hstack([initial_state_I, tf - tj])
    initial_state_P = Variable(torch.Tensor(initial_state_P))  # make it a torch variable

    for step_j in range(t_steps_CR):
        controls = policy_CR(initial_state_P)
        if plot_CR:
            action = select_action(controls[0], std_sqr, train=False)
        elif not plot_CR:
            action, log_prob_a, entropy = select_action(controls[0], std_sqr, train=True)
        control = {'U_u': np.float64(action)}
        final_state = model_integration(
            params, initial_state_I, control, dtime)
        if not plot_CR:
            log_probs_l[epi_n][step_j] = log_prob_a  # global var
        initial_state_I = copy.deepcopy(final_state)
        tj = tj + dtime  # calculate next time
        initial_state_P = np.hstack([initial_state_I, tf - tj])
        initial_state_P = Variable(torch.Tensor(initial_state_P)
                                   )  # make it a torch variable

        if plot_CR:
            y1_CR[step_j] = final_state[0]
            y2_CR[step_j] = final_state[1]
            t_CR[step_j] += tj
            U_u_CR[step_j] = np.float64(action)
    reward_CR = final_state[1]
    if plot_CR:
        return reward_CR, y1_CR, y2_CR, t_CR, U_u_CR
    if not plot_CR:
        return reward_CR
