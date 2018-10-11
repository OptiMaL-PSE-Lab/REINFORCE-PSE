import os
from os.path import join
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Normal, Beta, TransformedDistribution
from torch.distributions.transforms import AffineTransform

from integrator import model_integration
from plots import plot_state_policy_evol

eps = np.finfo(np.float32).eps.item()

def select_action(mean, sigma, lower_limit=0.0, upper_limit=5.0):
    """
    Find the required concentration hyperparameters in the canonical Beta distribution
    that will return the desired mean and deviation after the affine transformation.
    """
    width = upper_limit - lower_limit
    assert width > 0
    assert sigma < width**2 / 4

    canonical_mean = (mean - lower_limit) / width
    canonical_sigma = sigma / width**2

    alpha_plus_beta = ( canonical_mean * (1 - canonical_mean) / canonical_sigma ** 2 ) - 1
    alpha = canonical_mean * alpha_plus_beta
    beta = (1 - canonical_mean) * alpha_plus_beta

    canonical = Beta(alpha, beta)
    transformation = AffineTransform(loc = lower_limit, scale = width)
    transformed = TransformedDistribution(canonical, transformation)

    action = transformed.sample()
    log_prob = transformed.log_prob(action)
    return action, log_prob

def pretraining(policy, objective_actions, objective_deviation, model_specs,
                learning_rate, epochs):
    """Trains parametric policy model to resemble desired starting function."""

    assert objective_deviation > 0

    # training parameters
    criterion = nn.MSELoss(reduction='elementwise_mean')
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    objective_means = torch.tensor(objective_actions)
    objective_stds = torch.tensor(
        [objective_deviation for _ in model_specs['time_points']]
        )

    empty_list = [None for _ in model_specs['time_points']]

    states          = empty_list.copy()
    predictions     = empty_list.copy()
    uncertainties   = empty_list.copy()

    for epoch in range(epochs):
        t = model_specs['ti']  # define initial time at each episode
        integrated_state = model_specs['initial_state']
        for ind, _ in enumerate(model_specs['time_points']):

            time_left = model_specs['tf'] - t
            state = Tensor((*integrated_state, time_left)) # add time left to state
            mean, std = policy(state)

            states[ind]        = state
            predictions[ind]   = mean
            uncertainties[ind] = std

            model_specs['U'] = objective_actions[ind]

            integrated_state = model_integration(integrated_state, model_specs)

            t = t + model_specs['subinterval']

        predicted_controls      = torch.stack(predictions).squeeze()
        predicted_uncertainty   = torch.stack(uncertainties).squeeze()

        loss = criterion(objective_means, predicted_controls) + \
               criterion(objective_stds, predicted_uncertainty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch:", epoch, "\t loss:", loss.item())

    return None


def run_episode(policy, model_specs, track_evolution=False):
    """
    Compute a single episode given a policy.
    If track_evolution is True, return the evolution of y1, y2 and U,
    otherwise return the collection of rewards and log_probabilities of each state.
    """

    container = [None for i in model_specs['time_points']]

    if track_evolution:
        y1 = container.copy()
        y2 = container.copy()
        U = container.copy()
    else:
        log_probs = container.copy()

    # define initial conditions
    t = model_specs['ti']
    integrated_state = model_specs['initial_state']

    for ind, _ in enumerate(model_specs['time_points']):

        timed_state = Tensor((*integrated_state, model_specs['tf'] - t))
        mean, std = policy(timed_state)
        action, log_prob = select_action(mean, std)

        model_specs['U'] = action
        integrated_state = model_integration(integrated_state, model_specs)

        if track_evolution:
            y1[ind] = integrated_state[0]
            y2[ind] = integrated_state[1]
            U[ind]  = action
        else:
            log_probs[ind] = log_prob

        t = t + model_specs['subinterval']  # calculate next time

    if track_evolution:
        return y1, y2, U
    else:
        reward = integrated_state[1]
        return reward, log_probs

def sample_episodes(policy, optimizer, sample_size, model_specs):
    """
    Executes n-episodes under the current stochastic policy,
    gets an average of the reward and the summed log probabilities
    and use them to form the baselined loss function to optimize.
    """

    rewards          = [None for _ in range(sample_size)]
    summed_log_probs = [None for _ in range(sample_size)]

    log_prob_R = 0.0
    for epi in range(sample_size):
        reward, log_probs = run_episode(policy, model_specs)
        rewards[epi] = reward
        summed_log_probs[epi] = sum(log_probs)

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)

    for epi in reversed(range(sample_size)):
        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + eps)
        log_prob_R = log_prob_R - summed_log_probs[epi] * baselined_reward

    mean_log_prob_R = log_prob_R / sample_size
    return mean_log_prob_R, reward_mean, reward_std


def training(policy, optimizer, epochs, epoch_episodes, model_specs):
    """Run the full episodic training schedule."""

    # prepare directories for results
    os.makedirs('figures', exist_ok=True)
    os.makedirs('serializations', exist_ok=True)

    rewards_record = []
    rewards_std_record = []

    print(f"Training for {epochs} iterations of {epoch_episodes} sampled episodes each!")
    for epoch in range(epochs):

        # train policy over n-sample episode's mean log probability
        mean_log_prob, reward_mean, reward_std = sample_episodes(
            policy, optimizer, epoch_episodes, model_specs
        )
        optimizer.zero_grad()
        mean_log_prob.backward()
        optimizer.step()

        # store example episode reward
        rewards_record.append(reward_mean)
        rewards_std_record.append(reward_std)

        print('epoch:', epoch)
        print(f'mean reward: {reward_mean:.3} +- {reward_std:.2}')

        # save example episode evolution plot
        store_path = join('figures', f'profile_epoch_{epoch}_REINFORCE.png')

        y1, y2, U = run_episode(
            policy, model_specs, track_evolution=True
            )
        plot_state_policy_evol(
            model_specs['time_points'], y1, y2, U, show=False, store_path=store_path
            )

    # store trained policy
    torch.save(policy, join('serializations', 'policy.pt'))

    return rewards_record
