import os
from os.path import join
import copy

# import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Normal, Beta, TransformedDistribution
from torch.distributions.transforms import AffineTransform

from integrator import model_integration
from plots import plot_episode_states

eps = np.finfo(np.float32).eps.item()

def forge_distribution(mean, sigma, lower_limit=0.0, upper_limit=5.0):
    """
    Find the required concentration hyperparameters in the canonical Beta distribution
    that will return the desired mean and deviation after the affine transformation.
    """
    width = upper_limit - lower_limit
    assert width > 0
    assert sigma < eps + width / 2, f"invalid std: {sigma.item()}"

    canonical_mean = (mean - lower_limit) / width
    canonical_sigma = sigma / width**2

    alpha_plus_beta = ( canonical_mean * (1 - canonical_mean) / canonical_sigma ** 2 ) - 1
    alpha = canonical_mean * alpha_plus_beta
    beta = (1 - canonical_mean) * alpha_plus_beta

    canonical = Beta(alpha, beta)
    transformation = AffineTransform(loc = lower_limit, scale = width)
    transformed = TransformedDistribution(canonical, transformation)

    return transformed

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

def plot_policy_sample(policy, model_specs, objective=None, show=True, store_path=None):
    """Compute a single episode of the given policy and plot it."""

    container = [None for i in model_specs['time_points']]
    y1 = container.copy()
    y2 = container.copy()
    U = container.copy()

    # define initial conditions
    t = model_specs['ti']
    integrated_state = model_specs['initial_state']

    for ind, _ in enumerate(model_specs['time_points']):

        timed_state = Tensor((*integrated_state, model_specs['tf'] - t))
        mean, std = policy(timed_state)
        dist = forge_distribution(mean, std)
        action = dist.sample()

        model_specs['U'] = action
        integrated_state = model_integration(integrated_state, model_specs)

        y1[ind] = integrated_state[0]
        y2[ind] = integrated_state[1]
        U[ind]  = action

        t = t + model_specs['subinterval']

    plot_episode_states(
        model_specs['time_points'], y1, y2, U,
        show=show, store_path=store_path, objective=objective
        )

# @ray.remote
def run_episode(policy, model_specs, policy_old=None, epsilon=0.2):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # define initial conditions
    t = model_specs['ti']
    integrated_state = model_specs['initial_state']

    surrogate = 0.0
    for _ in model_specs['time_points']:

        timed_state = Tensor((*integrated_state, model_specs['tf'] - t))
        mean, std = policy(timed_state)
        dist = forge_distribution(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if policy_old is None:
            surrogate = surrogate + log_prob
        else:
            mean_old, std_old = policy_old(timed_state)
            dist_old = forge_distribution(mean_old, std_old)

            # probability of same action under older distribution
            log_prob_old = dist_old.log_prob(action).item() # avoid tracked gradients
            prob_ratio = (log_prob - log_prob_old).exp()
            clipped = prob_ratio.clamp(1-epsilon, i+epsilon)
            surrogate = surrogate + torch.min(prob_ratio, clipped)
            # NOTE: should be the min of the same factors multiplied by the advantage
            #       function. This is correct if the advantage function is positive...

        model_specs['U'] = action
        t = t + model_specs['subinterval']
        integrated_state = model_integration(integrated_state, model_specs)

    reward = integrated_state[1]
    return reward, surrogate

def sample_episodes(policy, sample_size, model_specs):
    """
    Executes n-episodes under the current stochastic policy,
    gets an average of the reward and the summed log probabilities
    and use them to form the baselined loss function to optimize.
    """

    rewards = [None for _ in range(sample_size)]
    surrogates = [None for _ in range(sample_size)]

    # NOTE: https://github.com/ray-project/ray/issues/2456
    # direct policy serialization loses the information of the tracked gradients...
    # samples = [run_episode.remote(policy, model_specs) for epi in range(sample_size)]

    log_prob_R = 0.0
    for epi in range(sample_size):
        # reward, surrogate = ray.get(samples[epi])
        reward, surrogate = run_episode(policy, model_specs)
        rewards[epi] = reward
        surrogates[epi] = surrogate

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)

    for epi in reversed(range(sample_size)):
        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + eps)
        log_prob_R = log_prob_R - surrogates[epi] * baselined_reward

    mean_log_prob_R = log_prob_R / sample_size
    return mean_log_prob_R, reward_mean, reward_std


def training(policy, optimizer, epochs, episode_batch, model_specs):
    """Run the full episodic training schedule."""

    # prepare directories for results
    os.makedirs('figures', exist_ok=True)
    os.makedirs('serializations', exist_ok=True)

    rewards_record = []
    rewards_std_record = []

    print(f"Training for {epochs} iterations of {episode_batch} sampled episodes each!")
    for epoch in range(epochs):

        # train policy over n-sample episode's mean log probability
        mean_log_prob, reward_mean, reward_std = sample_episodes(
            policy, episode_batch, model_specs
        )
        optimizer.zero_grad()
        mean_log_prob.backward()
        optimizer.step()

        # store example episode reward
        rewards_record.append(reward_mean)
        rewards_std_record.append(reward_std)

        print('epoch:', epoch)
        print(f'mean reward: {reward_mean:.3} +- {reward_std:.2}')

        # save sampled episode plot
        store_path = join('figures', f'profile_epoch_{epoch}_REINFORCE.png')
        plot_policy_sample(policy, model_specs, show=False, store_path=store_path)

    # store trained policy
    torch.save(policy, join('serializations', 'policy.pt'))

    return rewards_record
