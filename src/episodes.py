import numpy as np
import torch
from torch import Tensor

from distributions import sample_actions, retrieve_sum_log_prob
from tqdm import trange
from config import EPS


class EpisodeSampler:
    "Several algorithms to sample paths given a model and a policy."

    def __init__(self, model, policy, config):
        self.model = model
        self.policy = policy
        self.config = config

        self.recorder = {}
        self.recorder["states"] = np.zeros(
            shape=(config.episode_batch, config.divisions, model.states_dims)
        )
        self.recorder["controls"] = np.zeros(
            shape=(config.episode_batch, config.divisions, model.controls_dims)
        )
        self.recorder["rewards"] = np.zeros(shape=(config.episode_batch))

        self.episode_number = 0

    def record_state_control(self, states, controls, time_index):
        "Store the current state and controls of the episode."
        self.recorder["states"][self.episode_number, time_index, :] = states
        self.recorder["controls"][self.episode_number, time_index, :] = controls

    def record_reward(self, reward):
        "Store the reward of the episode"
        self.recorder["rewards"][self.episode_number] = reward

    def episode_reinforce(self):
        """Compute a single episode given a policy and track useful quantities for learning."""

        # define initial conditions
        t = self.config.ti
        current_state = self.config.initial_state

        sum_log_probs = 0.0
        hidden_state = None
        for time_index, _ in enumerate(self.config.time_points):

            timed_state = Tensor((*current_state, self.config.tf - t))
            (means, sigmas), hidden_state = self.policy(
                timed_state, hidden_state=hidden_state
            )
            controls, sum_log_prob = sample_actions(means, sigmas)

            sum_log_probs = sum_log_probs + sum_log_prob

            current_state = self.model.step(
                current_state, controls, self.config.subinterval, initial_time=t
            )
            t = t + self.config.subinterval

            self.record_state_control(current_state, controls, time_index)

        reward = current_state[1]
        self.record_reward(reward)

        self.episode_number += 1

        return reward, sum_log_probs

    def episode_ppo(self, policy_old=None):
        """Compute a single episode given a policy and track useful quantities for learning."""

        # define initial conditions
        t = self.config.ti
        current_state = self.config.initial_state

        prob_ratios = []
        hidden_state = None
        for time_index, _ in enumerate(self.config.time_points):

            timed_state = Tensor((*current_state, self.config.tf - t))
            (means, sigmas), hidden_state = self.policy(
                timed_state, hidden_state=hidden_state
            )
            controls, sum_log_prob = sample_actions(means, sigmas)

            # NOTE: probability of same action under older distribution
            #       avoid tracked gradients in old policy
            if policy_old is None or policy_old == self.policy:
                sum_log_prob_old = sum_log_prob
            else:
                means_old, sigmas_old = policy_old(timed_state)
                sum_log_prob_old = retrieve_sum_log_prob(
                    means_old, sigmas_old, controls
                )

            prob_ratio = (sum_log_prob - sum_log_prob_old).exp()
            prob_ratios.append(prob_ratio)

            current_state = self.model.step(
                current_state, controls, self.config.subinterval, initial_time=t
            )
            t = t + self.config.subinterval

            self.record_state_control(current_state, controls, time_index)

        reward = current_state[1]
        self.record_reward(reward)

        self.episode_number += 1

        return reward, prob_ratios

    def sample_episodes_reinforce(self):
        """
        Executes multiple episodes under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        rewards = [None for _ in range(self.config.episode_batch)]
        sum_log_probs = [None for _ in range(self.config.episode_batch)]

        for epi in trange(self.config.episode_batch, desc="Sampling episodes"):
            reward, sum_log_prob = self.episode_reinforce()
            rewards[epi] = reward
            sum_log_probs[epi] = sum_log_prob

        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)

        log_prob_R = 0.0
        for epi in reversed(range(self.config.episode_batch)):
            baselined_reward = (rewards[epi] - reward_mean) / (reward_std + EPS)
            log_prob_R = log_prob_R - sum_log_probs[epi] * baselined_reward

        mean_log_prob_R = log_prob_R / self.config.episode_batch
        return mean_log_prob_R, reward_mean, reward_std

    def sample_episodes_ppo(self, policy_old=None, epsilon=0.3):
        """
        Executes multiple episodes under the current stochastic policy,
        gets an average of the reward and the probabilities ratio between subsequent policies
        and use them to form the surrogate loss function to optimize.
        """

        rewards = [None for _ in range(self.config.episode_batch)]
        prob_ratios = [None for _ in range(self.config.episode_batch)]

        for epi in trange(self.config.episode_batch, desc="Sampling episodes"):
            reward, prob_ratios_episode = self.episode_ppo(policy_old=policy_old)
            rewards[epi] = reward
            prob_ratios[epi] = prob_ratios_episode

        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)

        surrogate = 0.0
        for epi in reversed(range(self.config.episode_batch)):

            baselined_reward = (rewards[epi] - reward_mean) / (reward_std + EPS)

            for prob_ratio in prob_ratios[epi]:
                clipped = prob_ratio.clamp(1 - epsilon, 1 + epsilon)
                # pylint: disable=no-member
                surrogate = surrogate - torch.min(
                    baselined_reward * prob_ratio, baselined_reward * clipped
                )

        mean_surrogate = surrogate / self.config.episode_batch
        return mean_surrogate, reward_mean, reward_std
