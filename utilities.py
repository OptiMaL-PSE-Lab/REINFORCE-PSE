import os
from os.path import join
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Normal

from integrator import model_integration
from plots import plot_state_policy_evol

eps = np.finfo(np.float32).eps.item()

def select_action(control_mean, control_sigma):
    """
    In the continuous space, this means adding a random perturbation to our control
    """
    dist = Normal(control_mean, control_sigma)
    control_choice = dist.rsample()
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
        for division in range(time_divisions):

            time_left = tf - t
            state = Tensor((*integrated_state, time_left)) # add time left to state

            states[division] = state
            controls[division] = policy(state)

            action = np.random.normal(loc=objective_actions[division], scale=pert_size)
            control_dict = {'U': np.float64(action)}

            integrated_state = model_integration(ode_params, integrated_state, control_dict, dtime)

            t = t + dtime

        input_controls = torch.stack(controls).squeeze()
        loss = criterion(objective_tensor, input_controls)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch:", epoch, "\t loss:", loss.item())

    return states, controls


def run_episode(policy, initial_state, ode_params, sigma,
                dtime, divisions, ti, tf, track_evolution=False):
    """
    Compute a single run given a policy.
    If track_evolution is True, return the evolution of y1, y2 and U,
    otherwise return the collection of rewards and log_probabilities of each state.
    """

    container = [None for i in range(divisions)]

    if track_evolution:
        y1 = container.copy()
        y2 = container.copy()
        U = container.copy()
    else:
        log_probs = container.copy()

    # define initial conditions
    t = ti
    integrated_state = initial_state

    for step in range(divisions):

        timed_state = Tensor((*integrated_state, tf - t))
        controls = policy(timed_state)
        action, log_prob, _ = select_action(controls, sigma)

        control = {'U': np.float64(action)}
        integrated_state = model_integration(ode_params, integrated_state, control, dtime)

        if track_evolution:
            y1[step] = integrated_state[0]
            y2[step] = integrated_state[1]
            U[step] = np.float64(action)
        else:
            log_probs[step] = log_prob

        t = t + dtime  # calculate next time
        timed_state = Tensor((*integrated_state, tf - t))

    if track_evolution:
        return y1, y2, U
    else:
        reward = integrated_state[1]
        return reward, log_probs

def sample_episodes(policy, optimizer, sample_size, sigma,
                    initial_state, ode_params, dtime, divisions, ti, tf):
    """
    Executes n-episodes to get an average of the reward multiplied by summed log probabilities
    that the current stochastic policy returns on each episode.
    """
    log_prob_R = 0.0
    rewards = [None for _ in range(sample_size)]
    summed_log_probs = [None for _ in range(sample_size)]

    for epi in range(sample_size):
        reward, log_probs = run_episode(
            policy, initial_state, ode_params, sigma,
            dtime, divisions, ti, tf
        )
        rewards[epi] = reward
        summed_log_probs[epi] = sum(log_probs)
    
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    
    for epi in reversed(range(sample_size)):
        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + eps)
        log_prob_R = log_prob_R - summed_log_probs[epi] * baselined_reward

    mean_log_prob_R = log_prob_R / sample_size
    return mean_log_prob_R


def training(policy, optimizer, epochs, epoch_episodes, sigma, sigma_reduction,
             ode_params, dtime, divisions, ti, tf):
    """Run the full episodic training schedule."""

    # prepare directories for results
    os.makedirs('figures', exist_ok=True)
    os.makedirs('serializations', exist_ok=True)

    rewards_record = []
    initial_state = np.array([1, 0])
    time_array = [ti + div * dtime for div in range(divisions)]

    for epoch in range(epochs):

        # train policy over n-sample episode's mean log probability
        mean_log_prob = sample_episodes(
            policy, optimizer, epoch_episodes, sigma,
            initial_state, ode_params, dtime, divisions, ti, tf
        )
        optimizer.zero_grad()
        mean_log_prob.backward()
        optimizer.step()

        # plot one episode of current state of policy
        y1, y2, U = run_episode(
            policy, initial_state, ode_params, sigma,
            dtime, divisions, ti, tf, track_evolution=True
            )

        # store example episode reward
        current_reward = y2[-1]
        rewards_record.append(current_reward)

        print('episode:', epoch)
        print('std_dev = ', round(sigma, 3))
        print('current_reward:', round(current_reward, 3))

        # save example episode evolution plot
        store_path = join('figures', f'profile_epoch_{epoch}_REINFORCE.png')
        plot_state_policy_evol(
            time_array, y1, y2, U, show=False, store_path=store_path
            )

        # reduce standard deviation
        sigma = sigma * sigma_reduction

    # store trained policy
    torch.save(policy, join('serializations', 'policy.pt'))

    return rewards_record
