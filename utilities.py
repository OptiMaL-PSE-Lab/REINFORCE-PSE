import sys
import os
from os.path import join
import copy
from numbers import Number

# import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from distributions import sample_actions, retrieve_sum_log_prob
from plots import plot_sampled_actions, plot_sampled_biactions, plot_reward_evolution

random_seed = np.random.randint(sys.maxsize) # maxsize = 2**63 - 1
torch.manual_seed(random_seed)
eps = np.finfo(np.float32).eps.item()

def iterable(controls):
    """Wrap control(s) in a iterable container."""
    if isinstance(controls, Number):
        return (controls,)
    return controls


def shift_grad_tracking(torch_object, track):
    """Set tracking flag for each parameter of torch object."""
    for param in torch_object.parameters():
        param.requires_grad = track


def pretraining(
    model,
    policy,
    objective_controls,
    objective_deviation,
    integration_config,
    learning_rate,
    iterations,
):
    """Trains parametric policy model to resemble desired starting function."""

    assert objective_deviation > 0

    num_controls = len(objective_controls[0])
    assert all(
        len(objective_control) == num_controls
        for objective_control in objective_controls
    )

    # training parameters
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # use tensors to track gradients
    objective_controls = torch.tensor(objective_controls)
    objective_deviations = torch.tensor(
        [
            (objective_deviation,) * num_controls
            for _ in integration_config["time_points"]
        ]
    )

    # predictions containers
    empty_list = [None for _ in integration_config["time_points"]]

    predictions = empty_list.copy()
    uncertainties = empty_list.copy()

    # iterative fitting
    for iteration in range(iterations):

        # define starting points at each episode
        t = integration_config["ti"]
        integrated_state = integration_config["initial_state"]

        # each step of this episode
        hidden_state = None
        for ind, _ in enumerate(integration_config["time_points"]):

            # current state tracked container
            time_left = integration_config["tf"] - t
            state = Tensor((*integrated_state, time_left))  # add time pending to state

            # continuous policy prediction
            (means, sigmas), hidden_state = policy(state, hidden_state=hidden_state)

            predictions[ind] = means
            uncertainties[ind] = sigmas

            # follow objective integration trajectory
            controls = iterable(objective_controls[ind])
            integration_time = integration_config["subinterval"]

            integrated_state = model.integrate(
                controls, integrated_state, integration_time, initial_time=t
            )
            t = t + integration_time

        # gather predictions of current policy
        predicted_controls = torch.stack(predictions)
        predicted_deviations = torch.stack(uncertainties)

        # optimize policy
        def closure():
            optimizer.zero_grad()
            loss = criterion(objective_controls, predicted_controls) + criterion(
                objective_deviations, predicted_deviations
            )
            print("iteration:", iteration, "\t loss:", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

def episode_reinforce(model, policy, integration_config, action_recorder=None):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # define initial conditions
    t = integration_config["ti"]
    integrated_state = integration_config["initial_state"]

    sum_log_probs = 0.0
    hidden_state = None
    for time_point in integration_config["time_points"]:

        timed_state = Tensor((*integrated_state, integration_config["tf"] - t))
        (means, sigmas), hidden_state = policy(timed_state, hidden_state=hidden_state)
        controls, sum_log_prob = sample_actions(means, sigmas)

        sum_log_probs = sum_log_probs + sum_log_prob

        integration_time = integration_config["subinterval"]

        t = t + integration_time
        integrated_state = model.integrate(
            controls, integrated_state, integration_time, initial_time=t
        )

        if action_recorder is not None:
            action_recorder[time_point].append(controls)  # FIXME?

    reward = integrated_state[1]
    return reward, sum_log_probs


# @ray.remote
def episode_ppo(
    model, policy, integration_config, policy_old=None, action_recorder=None
):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # define initial conditions
    t = integration_config["ti"]
    integrated_state = integration_config["initial_state"]

    prob_ratios = []
    hidden_state = None
    for time_point in integration_config["time_points"]:

        timed_state = Tensor((*integrated_state, integration_config["tf"] - t))
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

        integration_time = integration_config["subinterval"]

        t = t + integration_time
        integrated_state = model.integrate(
            controls, integrated_state, integration_time, initial_time=t
        )

        if action_recorder is not None:
            action_recorder[time_point].append(controls)

    reward = integrated_state[1]
    return reward, prob_ratios


def sample_episodes_reinforce(
    model, policy, sample_size, integration_config, action_recorder=None
):
    """
    Executes multiple episodes under the current stochastic policy,
    gets an average of the reward and the summed log probabilities
    and use them to form the baselined loss function to optimize.
    """

    rewards = [None for _ in range(sample_size)]
    sum_log_probs = [None for _ in range(sample_size)]

    for epi in range(sample_size):
        # reward, sum_log_prob = ray.get(samples[epi])
        reward, sum_log_prob = episode_reinforce(
            model, policy, integration_config, action_recorder=action_recorder
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


def sample_episodes_ppo(
    model,
    policy,
    sample_size,
    integration_config,
    policy_old=None,
    epsilon=0.3,
    action_recorder=None,
):
    """
    Executes multiple episodes under the current stochastic policy,
    gets an average of the reward and the probabilities ratio between subsequent policies
    and use them to form the surrogate loss function to optimize.
    """

    rewards = [None for _ in range(sample_size)]
    prob_ratios = [None for _ in range(sample_size)]

    for epi in range(sample_size):
        reward, prob_ratios_episode = episode_ppo(
            model,
            policy,
            integration_config,
            policy_old=policy_old,
            action_recorder=action_recorder,
        )
        rewards[epi] = reward
        prob_ratios[epi] = prob_ratios_episode

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)

    surrogate = 0.0
    for epi in reversed(range(sample_size)):

        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + eps)

        for prob_ratio in prob_ratios[epi]:
            clipped = prob_ratio.clamp(1 - epsilon, 1 + epsilon)
            surrogate = surrogate - torch.min(
                baselined_reward * prob_ratio, baselined_reward * clipped
            )

    mean_surrogate = surrogate / sample_size
    return mean_surrogate, reward_mean, reward_std


def training(
    model, policy, integration_config, optim_config, record_graphs=False, model_id=None
):
    """Run the full episodic training schedule."""

    assert model_id != None, "please provide the model_id keyword"
    assert (
        optim_config["method"] == "reinforce" or optim_config["method"] == "ppo"
    ), "methods supported: reinforce and ppo"

    # prepare directories for results
    os.makedirs("serializations", exist_ok=True)
    if record_graphs:
        plots_dir = join(
            "figures",
            (
                f"policy_{policy.__class__.__name__}_"
                f"method_{optim_config['method']}_"
                f"batch_{optim_config['episode_batch']}_"
                f"iter_{optim_config['iterations']}"
            ),
        )
        os.makedirs(plots_dir, exist_ok=True)  # FIXME: remove exists_ok

    reward_recorder = []
    rewards_std_record = []

    if record_graphs:
        action_recorder = {
            time_point: [] for time_point in integration_config["time_points"]
        }
    else:
        action_recorder = None

    print(
        f"""
        Training for {optim_config['iterations']} iterations of
        {optim_config['episode_batch']} sampled episodes each!
        """
    )

    optimizer = optim.Adam(policy.parameters(), lr=optim_config["learning_rate"])

    for iteration in range(optim_config["iterations"]):

        if optim_config["method"] == "reinforce":
            surrogate_mean, reward_mean, reward_std = sample_episodes_reinforce(
                model,
                policy,
                optim_config["episode_batch"],
                integration_config,
                action_recorder=action_recorder,
            )
        elif optim_config["method"] == "ppo":
            if iteration == 0:
                policy_old = None

            surrogate_mean, reward_mean, reward_std = sample_episodes_ppo(
                model,
                policy,
                optim_config["episode_batch"],
                integration_config,
                policy_old=policy_old,
                action_recorder=action_recorder,
            )

        # maximize expected surrogate function
        if optim_config["method"] == "ppo":
            policy_old = copy.deepcopy(policy)

        for _ in range(optim_config["epochs"]):

            optimizer.zero_grad()  # FIXME: should this be outside of the loop??
            surrogate_mean.backward(retain_graph=True)
            optimizer.step()

        # store mean episode reward
        reward_recorder.append(reward_mean)
        rewards_std_record.append(reward_std)

        print("iteration:", iteration)
        print(f"mean reward: {reward_mean:.5} +- {reward_std:.4}")

        if record_graphs:

            plot_name = (
                f"action_distribution_"
                f"id_{model_id}_"
                f"lr_{optim_config['learning_rate']}_"
                f"iteration_{iteration:03d}.png"
            )
            if model.controls_dims == 2:
                plot_sampled_biactions(
                    action_recorder,
                    iteration,
                    show=False,
                    store_path=join(plots_dir, plot_name),
                )
            else:
                plot_sampled_actions(
                    action_recorder,
                    iteration,
                    show=False,
                    store_path=join(plots_dir, plot_name),
                )

    # NOTE: separated to have all rewards accesible to tune ylims accordingly
    if record_graphs:
        for iteration in range(optim_config["iterations"]):
            plot_name = (
                f"reward_"
                f"id_{model_id}_"
                f"lr_{optim_config['learning_rate']}_"
                f"iteration_{iteration:03d}.png"
            )
            plot_reward_evolution(
                reward_recorder,
                iteration,
                optim_config,
                show=False,
                store_path=join(plots_dir, plot_name),
            )

    # store trained policy
    torch.save(policy.state_dict(), join("serializations", f"{model_id}_policy.pt"))
