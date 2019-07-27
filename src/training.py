import sys
import copy

# import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from utils import iterable, FIGURES_DIR, POLICIES_DIR
from episodes import sample_episodes_ppo, sample_episodes_reinforce
from plots import plot_sampled_actions, plot_sampled_biactions, plot_reward_evolution

random_seed = np.random.randint(sys.maxsize)  # maxsize = 2**63 - 1
torch.manual_seed(random_seed)


def pretrainer(
    model,
    policy,
    objective_controls,
    objective_deviation,
    config,
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
    optimizer = optim.Adam(policy.parameters(), lr=config.pre_learning_rate)

    # use tensors to track gradients
    objective_controls = torch.tensor(objective_controls)
    objective_deviations = torch.tensor(
        [
            (objective_deviation,) * num_controls
            for _ in config.time_points
        ]
    )

    # predictions containers
    empty_list = [None for _ in config.time_points]

    predictions = empty_list.copy()
    uncertainties = empty_list.copy()

    # iterative fitting
    for iteration in range(config.pre_iterations):

        # define starting points at each episode
        t = config.ti
        integrated_state = config.initial_state

        # each step of this episode
        hidden_state = None
        for ind, _ in enumerate(config.time_points):

            # current state tracked container
            time_left = config.tf - t
            state = Tensor((*integrated_state, time_left))  # add time pending to state

            # continuous policy prediction
            (means, sigmas), hidden_state = policy(state, hidden_state=hidden_state)

            predictions[ind] = means
            uncertainties[ind] = sigmas

            # follow objective integration trajectory
            controls = iterable(objective_controls[ind])
            integration_time = config.subinterval

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


def trainer(model, policy, config):
    """Run the full episodic training schedule."""

    # prepare directories for results
    if not config.discard_graphics:
        plots_dir = FIGURES_DIR / (
            f"policy_{policy.__class__.__name__}_"
            f"method_{config.policy_gradient_method}_"
            f"batch_{config.episode_batch}_"
            f"iter_{config.iterations}"
        )
        plots_dir.mkdir()

    reward_recorder = []
    rewards_std_record = []

    if not config.discard_graphics:
        action_recorder = {
            time_point: [] for time_point in config.time_points
        }
    else:
        action_recorder = None

    print(
        f"""
        Training for {config.iterations} iterations of
        {config.episode_batch} sampled episodes each!
        """
    )

    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

    for iteration in range(config.iterations):

        if config.policy_gradient_method == "reinforce":
            surrogate_mean, reward_mean, reward_std = sample_episodes_reinforce(
                model,
                policy,
                config,
                action_recorder=action_recorder,
            )
        elif config.policy_gradient_method == "ppo":
            if iteration == 0:
                policy_old = None

            surrogate_mean, reward_mean, reward_std = sample_episodes_ppo(
                model,
                policy,
                config,
                policy_old=policy_old,
                action_recorder=action_recorder,
            )

        # maximize expected surrogate function
        if config.policy_gradient_method == "ppo":
            policy_old = copy.deepcopy(policy)

        for _ in range(config.chained_steps):

            optimizer.zero_grad()  # FIXME: should this be outside of the loop??
            surrogate_mean.backward(retain_graph=True)
            optimizer.step()

        # store mean episode reward
        reward_recorder.append(reward_mean)
        rewards_std_record.append(reward_std)

        print("iteration:", iteration)
        print(f"mean reward: {reward_mean:.5} +- {reward_std:.4}")

        if not config.discard_graphics:

            plot_name = (
                f"action_distribution_"
                f"model_{model.__class__.__name__}_"
                f"lr_{config.learning_rate}_"
                f"iteration_{iteration:03d}.png"
            )
            if model.controls_dims == 2:
                plot_sampled_biactions(
                    action_recorder,
                    iteration,
                    show=False,
                    store_path=plots_dir / plot_name,
                )
            else:
                plot_sampled_actions(
                    action_recorder,
                    iteration,
                    show=False,
                    store_path=plots_dir / plot_name,
                )

    # NOTE: separated to have all rewards accesible to tune ylims accordingly
    if not config.discard_graphics:
        for iteration in range(config.iterations):
            plot_name = (
                f"reward_"
                f"model_{model.__class__.__name__}_"
                f"lr_{config.learning_rate}_"
                f"iteration_{iteration:03d}.png"
            )
            plot_reward_evolution(
                reward_recorder,
                iteration,
                config,
                show=False,
                store_path=plots_dir / plot_name,
            )

    # store trained policy
    torch.save(policy.state_dict(), POLICIES_DIR / f"{model.__class__.__name__}_policy.pt")
