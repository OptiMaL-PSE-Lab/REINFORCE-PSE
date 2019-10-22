import copy
import sys
from pathlib import PurePath

import h5py
import numpy as np
import torch
from torch import Tensor
from tqdm import trange  # trange = tqdm(range(...))

from utils import iterable_container
from episodes import EpisodeSampler
from policies import policy_selector


class Trainer:
    "Training utilities."

    def __init__(self, model, config):

        self.seed = np.random.randint(sys.maxsize)  # maxsize = 2**63 - 1
        print(f"Using torch random seed {self.seed}!")
        torch.manual_seed(self.seed)

        self.config = config
        self.policy = policy_selector(model, config)

        self.set_model(model)

    def set_model(self, model):
        "Sets all fields related to the model"
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.policy_file = self.config.policies_dir / f"{self.model_name}.pt"

    def pretrain(self, objective_controls, objective_deviation):
        """Trains parametric policy model to resemble desired starting function."""

        assert objective_deviation > 0

        num_controls = len(objective_controls[0])
        assert all(
            len(objective_control) == num_controls
            for objective_control in objective_controls
        )

        # training parameters
        criterion = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.pre_learning_rate
        )

        # use tensors to track gradients
        # pylint: disable=not-callable  # https://github.com/pytorch/pytorch/issues/24807
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
                controls = iterable_container(objective_controls[ind])

                current_state = self.model.step(
                    current_state, controls, self.config.subinterval, initial_time=t
                )
                t = t + self.config.subinterval

            def closure():

                # gather predictions of current policy
                predicted_controls = torch.stack(predictions)
                predicted_deviations = torch.stack(uncertainties)

                # optimize policy
                optimizer.zero_grad()
                loss = criterion(objective_controls, predicted_controls) + criterion(
                    objective_deviations, predicted_deviations
                )
                pbar.set_description(f"Pretraining loss {loss.item():.2}")
                loss.backward()
                return loss

            optimizer.step(closure)

    def train(self, post_training=False):
        """Run the full episodic training schedule."""

        try:  # check configuration consistency with target data file
            with h5py.File(self.config.data_file, mode="r") as h5file:
                for key, val in self.config.__dict__.items():
                    stored_val = h5file.attrs[key]
                    if isinstance(stored_val, np.ndarray):
                        if np.array_equal(stored_val, val):
                            continue
                    if isinstance(val, PurePath):
                        if str(val) == stored_val:
                            continue
                    if h5file.attrs[key] == val:
                        continue
                    raise RuntimeWarning(
                        f"Non matching config {key}!\n"
                        f"Current: {val}\n"
                        f"Stored: {stored_val}!"
                    )
        except OSError:  # non-existent file
            # store current configuration
            with h5py.File(self.config.data_file) as h5file:  # mode="a"
                for key, val in self.config.__dict__.items():
                    if isinstance(val, PurePath):
                        val = str(val)
                    h5file.attrs[key] = val

        if post_training:
            iterations = self.config.post_iterations
            learning_rate = self.config.post_learning_rate
        else:
            iterations = self.config.iterations
            learning_rate = self.config.learning_rate

        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        pbar = trange(iterations)
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

            with h5py.File(self.config.data_file) as h5file:  # mode="a"

                # NOTE: h5py does not support unicode directly
                model_ascii = np.string_(self.model_name)
                seeds = h5file.attrs.get("seeds", np.array([], dtype=int))
                models = h5file.attrs.get("models", np.array([], dtype="S"))

                if self.seed not in seeds:
                    h5file.attrs["seeds"] = np.append(seeds, self.seed)
                if model_ascii not in models:
                    h5file.attrs["models"] = np.append(models, model_ascii)

                model_group = h5file.require_group(f"model_{self.model_name}")
                model_group.attrs["iterations"] = iterations
                model_group.attrs["learning_rate"] = learning_rate

                iter_group = model_group.require_group(f"iter_{iteration}")
                states_group = iter_group.require_group("states")
                controls_group = iter_group.require_group("controls")
                rewards_group = iter_group.require_group("rewards")

                # gzip with given level of compression (0-9)
                seed_flag = f"seed_{self.seed}"
                states_group.create_dataset(
                    seed_flag, data=ep_sampler.recorder["states"], compression=9
                )
                controls_group.create_dataset(
                    seed_flag, data=ep_sampler.recorder["controls"], compression=9
                )
                rewards_group.create_dataset(
                    seed_flag, data=ep_sampler.recorder["rewards"], compression=9
                )

            pbar.set_description(
                f"Roll-out mean reward: {reward_mean:.3} +- {reward_std:.2}"
            )

        # store trained policy
        torch.save(self.policy.state_dict(), self.policy_file)
