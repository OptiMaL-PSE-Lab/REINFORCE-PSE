import numpy as np
import torch
from torch import Tensor

from utils import EPS
from distributions import sample_actions, retrieve_sum_log_prob


def episode_reinforce(model, policy, config, action_recorder=None):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # define initial conditions
    t = config.ti
    current_state = config.initial_state

    sum_log_probs = 0.0
    hidden_state = None
    for time_point in config.time_points:

        timed_state = Tensor((*current_state, config.tf - t))
        (means, sigmas), hidden_state = policy(timed_state, hidden_state=hidden_state)
        controls, sum_log_prob = sample_actions(means, sigmas)

        sum_log_probs = sum_log_probs + sum_log_prob

        current_state = model.step(
            current_state, controls, config.subinterval, initial_time=t
        )
        t = t + config.subinterval

        if action_recorder is not None:
            action_recorder[time_point].append(controls)  # FIXME?

    reward = current_state[1]
    return reward, sum_log_probs


# @ray.remote
def episode_ppo(model, policy, config, policy_old=None, action_recorder=None):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # define initial conditions
    t = config.ti
    current_state = config.initial_state

    prob_ratios = []
    hidden_state = None
    for time_point in config.time_points:

        timed_state = Tensor((*current_state, config.tf - t))
        (means, sigmas), hidden_state = policy(timed_state, hidden_state=hidden_state)
        controls, sum_log_prob = sample_actions(means, sigmas)

        # NOTE: probability of same action under older distribution
        #       avoid tracked gradients in old policy
        if policy_old is None or policy_old == policy:
            sum_log_prob_old = sum_log_prob
        else:
            means_old, sigmas_old = policy_old(timed_state)
            sum_log_prob_old = retrieve_sum_log_prob(means_old, sigmas_old, controls)

        prob_ratio = (sum_log_prob - sum_log_prob_old).exp()
        prob_ratios.append(prob_ratio)

        current_state = model.step(
            current_state, controls, config.subinterval, initial_time=t
        )
        t = t + config.subinterval

        if action_recorder is not None:
            action_recorder[time_point].append(controls)

    reward = current_state[1]
    return reward, prob_ratios


def sample_episodes_reinforce(model, policy, config, action_recorder=None):
    """
    Executes multiple episodes under the current stochastic policy,
    gets an average of the reward and the summed log probabilities
    and use them to form the baselined loss function to optimize.
    """

    rewards = [None for _ in range(config.episode_batch)]
    sum_log_probs = [None for _ in range(config.episode_batch)]

    for epi in range(config.episode_batch):
        # reward, sum_log_prob = ray.get(samples[epi])
        reward, sum_log_prob = episode_reinforce(
            model, policy, config, action_recorder=action_recorder
        )
        rewards[epi] = reward
        sum_log_probs[epi] = sum_log_prob

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)

    log_prob_R = 0.0
    for epi in reversed(range(config.episode_batch)):
        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + EPS)
        log_prob_R = log_prob_R - sum_log_probs[epi] * baselined_reward

    mean_log_prob_R = log_prob_R / config.episode_batch
    return mean_log_prob_R, reward_mean, reward_std


def sample_episodes_ppo(
    model, policy, config, policy_old=None, epsilon=0.3, action_recorder=None
):
    """
    Executes multiple episodes under the current stochastic policy,
    gets an average of the reward and the probabilities ratio between subsequent policies
    and use them to form the surrogate loss function to optimize.
    """

    rewards = [None for _ in range(config.episode_batch)]
    prob_ratios = [None for _ in range(config.episode_batch)]

    for epi in range(config.episode_batch):
        reward, prob_ratios_episode = episode_ppo(
            model,
            policy,
            config,
            policy_old=policy_old,
            action_recorder=action_recorder,
        )
        rewards[epi] = reward
        prob_ratios[epi] = prob_ratios_episode

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)

    surrogate = 0.0
    for epi in reversed(range(config.episode_batch)):

        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + EPS)

        for prob_ratio in prob_ratios[epi]:
            clipped = prob_ratio.clamp(1 - epsilon, 1 + epsilon)
            surrogate = surrogate - torch.min(
                baselined_reward * prob_ratio, baselined_reward * clipped
            )

    mean_surrogate = surrogate / config.episode_batch
    return mean_surrogate, reward_mean, reward_std
