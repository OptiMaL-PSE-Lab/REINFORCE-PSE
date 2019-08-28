import sys
import copy
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import trange  # trange = tqdm(range(...))

from initial_controls import multilabel_cheby_identifiers
from utils import iterable, DIRS
from episodes import EpisodeSampler
from policies import policy_selector
from plots import plot_sampled_actions, plot_sampled_biactions, plot_reward_evolution

class Trainer:
    "Training utilities."

    def __init__(self, model, config):

        self.seed = np.random.randint(sys.maxsize)  # maxsize = 2**63 - 1
        print(f"Using random seed {self.seed}!")
        torch.manual_seed(self.seed)

        self.model = model
        self.config = config
        self.policy = policy_selector(model, config)

        self.filename = f"policy_{self.policy.__class__.__name__}.hdf5"

    def pretrain(self, objective_controls, objective_deviation):
        """Trains parametric policy model to resemble desired starting function."""

        self.filename = (
            f"policy_{self.policy.__class__.__name__}_"
            f"initial_controls_{self.config.initial_controls_labels}"
            ".hdf5"
        )

        assert objective_deviation > 0

        num_controls = len(objective_controls[0])
        assert all(
            len(objective_control) == num_controls
            for objective_control in objective_controls
        )

        # training parameters
        criterion = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(
            self.policy.parameters(), lr=self.config.pre_learning_rate
        )

        # use tensors to track gradients
        objective_controls = torch.tensor(objective_controls)
        objective_deviations = torch.tensor(
            [(objective_deviation,) * num_controls for _ in self.config.time_points]
        )

        # predictions containers
        empty_list = [None for _ in self.config.time_points]

        predictions = empty_list.copy()
        uncertainties = empty_list.copy()

        # iterative fitting
        pbar = trange(self.config.pre_iterations)
        for _ in pbar:

            # define starting points at each episode
            t = self.config.ti
            current_state = self.config.initial_state

            # each step of this episode
            hidden_state = None
            for ind, _ in enumerate(self.config.time_points):

                # current state tracked container
                time_left = self.config.tf - t
                state = Tensor((*current_state, time_left))  # add time pending to state

                # continuous policy prediction
                (means, sigmas), hidden_state = self.policy(
                    state, hidden_state=hidden_state
                )

                predictions[ind] = means
                uncertainties[ind] = sigmas

                # follow objective integration trajectory
                controls = iterable(objective_controls[ind])

                current_state = self.model.step(
                    current_state, controls, self.config.subinterval, initial_time=t
                )
                t = t + self.config.subinterval

            # gather predictions of current policy
            predicted_controls = torch.stack(predictions)
            predicted_deviations = torch.stack(uncertainties)

            # optimize policy
            def closure():
                optimizer.zero_grad()
                loss = criterion(objective_controls, predicted_controls) + criterion(
                    objective_deviations, predicted_deviations
                )
                pbar.set_description(f"Pretraining loss {loss.item():.2}")
                loss.backward()
                return loss

            optimizer.step(closure)

    def train(self):
        """Run the full episodic training schedule."""

        recorder = {
            "rewards_mean": np.zeros(shape=(self.config.iterations)),
            "rewards_std": np.zeros(shape=(self.config.iterations)),
        }

        # # prepare directories for results
        # if not config.discard_graphics:
        #     chebyshev_labels = config.initial_controls_labels
        #     plots_dir = DIRS["figures"] / (
        #         f"policy_{policy.__class__.__name__}_"
        #         f"method_{config.policy_gradient_method}_"
        #         f"batch_{config.episode_batch}_"
        #         f"controls_{chebyshev_labels}"
        #     )
        #     plots_dir.mkdir(exist_ok=True)

        # print(
        #     f"""
        #     Training for {self.config.iterations} iterations of
        #     {self.config.episode_batch} sampled episodes each!
        #     """
        # )

        optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

        pbar = trange(self.config.iterations)
        for iteration in pbar:

            ep_sampler = EpisodeSampler(self.model, self.policy, self.config)

            if self.config.policy_gradient_method == "reinforce":
                surrogate_mean, reward_mean, reward_std = (
                    ep_sampler.sample_episodes_reinforce()
                )
            elif self.config.policy_gradient_method == "ppo":
                if iteration == 0:
                    policy_old = None
                surrogate_mean, reward_mean, reward_std = ep_sampler.sample_episodes_ppo(
                    policy_old=policy_old
                )

            # maximize expected surrogate function
            if self.config.policy_gradient_method == "ppo":
                policy_old = copy.deepcopy(self.policy)

            for _ in range(self.config.chained_steps):

                optimizer.zero_grad()  # FIXME: should this be outside of the loop??
                surrogate_mean.backward(retain_graph=True)
                optimizer.step()

            # store mean episode reward
            recorder["rewards_mean"][iteration] = reward_mean
            recorder["rewards_std"][iteration] = reward_std

            with h5py.File(DIRS["data"] / self.filename) as h5file:  # mode="a"
                if len(h5file.attrs) == 0:
                    for key, val in self.config.__dict__.items():
                        h5file.attrs[key] = val

                group = h5file.create_group(
                    f"seed_{self.seed}/model_{self.model.__class__.__name__}/iter_{iteration}"
                )
                group["states"] = ep_sampler.recorder["states"]
                group["controls"] = ep_sampler.recorder["controls"]
                group["rewards"] = ep_sampler.recorder["rewards"]

            pbar.set_description(f"mean reward: {reward_mean:.3} +- {reward_std:.2}")

        #     if not config.discard_graphics:

        #         plot_name = (
        #             f"action_distribution_"
        #             f"model_{model.__class__.__name__}_"
        #             f"lr_{config.learning_rate}_"
        #             f"iteration_{iteration:03d}.png"
        #         )
        #         if model.controls_dims == 2:
        #             plot_sampled_biactions(
        #                 action_recorder,
        #                 iteration,
        #                 show=False,
        #                 store_path=plots_dir / plot_name,
        #             )
        #         else:
        #             plot_sampled_actions(
        #                 action_recorder,
        #                 iteration,
        #                 show=False,
        #                 store_path=plots_dir / plot_name,
        #             )

        # # NOTE: separated to have all rewards accesible to tune ylims accordingly
        # if not config.discard_graphics:
        #     for iteration in range(config.iterations):
        #         plot_name = (
        #             f"reward_"
        #             f"model_{model.__class__.__name__}_"
        #             f"lr_{config.learning_rate}_"
        #             f"iteration_{iteration:03d}.png"
        #         )
        #         # TODO: plot reward with std
        #         plot_reward_evolution(
        #             recorder["rewards_mean"],
        #             iteration,
        #             config,
        #             show=False,
        #             store_path=plots_dir / plot_name,
        #         )

        # store trained policy
        torch.save(
            self.policy.state_dict(),
            DIRS["results"] / "policies" / f"{self.model.__class__.__name__}_policy.pt"
        )
