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
from plots import plot_episode_states, plot_sampled_actions

eps = np.finfo(np.float32).eps.item()

def forge_distribution(mean, sigma, lower_limit=0.0, upper_limit=5.0):
    """
    Find the required concentration hyperparameters in the canonical Beta distribution
    that will return the desired mean and deviation after the affine transformation.
    """
    width = upper_limit - lower_limit
    assert width > 0
    assert sigma <= eps + width / 2, f"invalid std: {sigma.item()}"

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
                learning_rate, iterations):
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

    for iteration in range(iterations):
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

        print("iteration:", iteration, "\t loss:", loss.item())

    return None

def plot_episode(policy, model_specs, objective=None, show=True, store_path=None):
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
def episode_reinforce(policy, model_specs, action_recorder=None):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # define initial conditions
    t = model_specs['ti']
    integrated_state = model_specs['initial_state']

    sum_log_probs = 0.0
    for time_point in model_specs['time_points']:

        timed_state = Tensor((*integrated_state, model_specs['tf'] - t))
        mean, std = policy(timed_state)
        dist = forge_distribution(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        sum_log_probs = sum_log_probs + log_prob

        model_specs['U'] = action
        t = t + model_specs['subinterval']
        integrated_state = model_integration(integrated_state, model_specs)

        if action_recorder is not None:
            action_recorder[time_point].append(action.item())

    reward = integrated_state[1]
    return reward, sum_log_probs

# @ray.remote
def episode_ppo(policy, model_specs, policy_old=None, action_recorder=None):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # define initial conditions
    t = model_specs['ti']
    integrated_state = model_specs['initial_state']

    prob_ratios = []
    for time_point in model_specs['time_points']:

        timed_state = Tensor((*integrated_state, model_specs['tf'] - t))
        mean, std = policy(timed_state)
        dist = forge_distribution(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # NOTE: probability of same action under older distribution
        #       avoid tracked gradients in old policy
        if policy_old is None or policy_old == policy:
            log_prob_old = log_prob.item()
        else:
            mean_old, std_old = policy_old(timed_state)
            dist_old = forge_distribution(mean_old, std_old)
            log_prob_old = dist_old.log_prob(action).item()

        prob_ratio = (log_prob - log_prob_old).exp()
        prob_ratios.append(prob_ratio)

        model_specs['U'] = action
        t = t + model_specs['subinterval']
        integrated_state = model_integration(integrated_state, model_specs)

        if action_recorder is not None:
            action_recorder[time_point].append(action.item())

    reward = integrated_state[1]
    return reward, prob_ratios

def sample_episodes_reinforce(policy, sample_size, model_specs, action_recorder=None):
    """
    Executes multiple episodes under the current stochastic policy,
    gets an average of the reward and the summed log probabilities
    and use them to form the baselined loss function to optimize.
    """

    rewards = [None for _ in range(sample_size)]
    sum_log_probs = [None for _ in range(sample_size)]

    # NOTE: https://github.com/ray-project/ray/issues/2456
    # direct policy serialization loses the information of the tracked gradients...
    # samples = [run_episode.remote(policy, model_specs) for epi in range(sample_size)]

    for epi in range(sample_size):
        # reward, sum_log_prob = ray.get(samples[epi])
        reward, sum_log_prob = episode_reinforce(
            policy, model_specs, action_recorder=action_recorder
            )
        rewards[epi] = reward
        sum_log_probs[epi] = sum_log_prob

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)

    log_prob_R = 0.0
    for epi in reversed(range(sample_size)):
        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + eps)
        log_prob_R = log_prob_R - sum_log_probs[epi] * baselined_reward

    mean_log_prob_R = log_prob_R / sample_size
    return mean_log_prob_R, reward_mean, reward_std

def sample_episodes_ppo(policy, sample_size, model_specs,
                        policy_old=None, epsilon=0.3, action_recorder=None):
    """
    Executes multiple episodes under the current stochastic policy,
    gets an average of the reward and the probabilities ratio between subsequent policies
    and use them to form the surrogate loss function to optimize.
    """

    rewards = [None for _ in range(sample_size)]
    prob_ratios = [None for _ in range(sample_size)]

    for epi in range(sample_size):
        reward, prob_ratios_episode = episode_ppo(
            policy, model_specs, policy_old=policy_old, action_recorder=action_recorder
            )
        rewards[epi] = reward
        prob_ratios[epi] = prob_ratios_episode

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)

    surrogate = 0.0
    for epi in reversed(range(sample_size)):

        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + eps)

        for prob_ratio in prob_ratios[epi]:
            clipped = prob_ratio.clamp( 1 - epsilon, 1 + epsilon )
            surrogate = surrogate - torch.min(
                prob_ratio * baselined_reward,
                clipped * baselined_reward
            )

    mean_surrogate = surrogate / sample_size
    return mean_surrogate, reward_mean, reward_std

def training(policy, optimizer, iterations, episode_batch, model_specs,
             method='reinforce', epochs=1, record_actions=False):
    """Run the full episodic training schedule."""

    assert method == 'reinforce' or method == 'ppo', "methods supported: reinforce and ppo"

    # prepare directories for results
    os.makedirs('figures', exist_ok=True)
    os.makedirs('serializations', exist_ok=True)

    rewards_record = []
    rewards_std_record = []

    if record_actions:
        action_recorder = {time_point: [] for time_point in model_specs['time_points']}
    else:
        action_recorder = None

    print(f"Training for {iterations} iterations of {episode_batch} sampled episodes each!")
    for iteration in range(iterations):

        if method == 'reinforce':
            surrogate_mean, reward_mean, reward_std = sample_episodes_reinforce(
                policy, episode_batch, model_specs, action_recorder=action_recorder
            )
        elif method == 'ppo':
            if iteration == 0:
                policy_old = None
            surrogate_mean, reward_mean, reward_std = sample_episodes_ppo(
                policy, episode_batch, model_specs,
                policy_old=policy_old, action_recorder=action_recorder
            )

        # maximize expected surrogate function
        if method == 'ppo':
            policy_old = copy.deepcopy(policy)

        for _ in range(epochs):
            optimizer.zero_grad()
            surrogate_mean.backward(retain_graph=True)
            optimizer.step()

        # store mean episode reward
        rewards_record.append(reward_mean)
        rewards_std_record.append(reward_std)

        print('iteration:', iteration)
        print(f'mean reward: {reward_mean:.3} +- {reward_std:.2}')

        # save sampled episode plot
        store_path = join('figures', f'profile_iteration_{iteration}_method_{method}.png')
        plot_episode(policy, model_specs, show=False, store_path=store_path)

        if record_actions:
            store_path = join(
                'figures',
                f'action_distribution_iteration_{iteration}_method_{method}.png'
                )
            plot_sampled_actions(
                action_recorder, iteration,
                show=False, store_path=store_path
                )

    # store trained policy
    torch.save(policy, join('serializations', 'policy_reinforce.pt'))

    return rewards_record
