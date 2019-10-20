from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

sns.set_style(style="whitegrid")

mpl.rc("font", size=SMALL_SIZE)  # controls default text sizes
mpl.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
mpl.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
mpl.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize

mpl.rc("figure", figsize=(6, 2))
mpl.rc("savefig", bbox="tight", dpi=600)

class Plotter:
    "Plotting utilities from data files."

    def __init__(self, config):

        self.config = config
        self.filepath = config.data_dir

        # FIXME: take both models into account
        with h5py.File(self.filepath, mode = "r") as h5file:
            self.seeds = h5file.attrs["seeds"]
            self.models = h5file.attrs["models"]

    @staticmethod
    def dataset_yield(name, node):
        if isinstance(node, h5py.Dataset):
            yield node

    @staticmethod
    def dataset_flag(name, node):
        if isinstance(node, h5py.Dataset):
            return 1
        return 0

    def dataset_iterator(self, h5file, model, dset_name):
        "Iterate over each iteration & seed keeping model constant"
        assert model in h5file.attrs["models"], f"Available models are {self.models}"
        seeds = h5file.attrs["seeds"]
        # FIXME 
        first_group = h5file[f"seed_{seeds[0]}/model_{model.__class__.__name__}"]
        iterations = sum(first_group.visititems(self.dataset_flag))
        datasets = []
        for iteration in range(iterations):
            for seed in seeds:
                group = h5file[(
                    f"seed_{seed}/"
                    f"model_{model.__class__.__name__}/"
                    f"iter_{iteration}"
                )]
                datasets.append(group[dset_name])
        return np.concatenate(datasets, axis=0)

    def plot_sampled_biactions(self, action_recorder, iteration, show=True, store_path=None):

        time_points = sorted(list(action_recorder.keys()))

        num_controls = len(action_recorder[time_points[0]][0])
        assert num_controls == 2, "This plotting routine is just intended for 2 controls."

        long_form_dict = {key: [] for key in ["time", "value", "control"]}

        for time_point in time_points:

            recorded_controls = action_recorder[time_point]

            for controls in recorded_controls:

                long_form_dict["time"].append(f"{time_point:.2f}")
                long_form_dict["value"].append(controls[0].item())
                long_form_dict["control"].append("control_0")

                long_form_dict["time"].append(f"{time_point:.2f}")
                long_form_dict["value"].append(controls[1].item())
                long_form_dict["control"].append("control_1")

        ax = sns.violinplot(
            x=long_form_dict["time"],
            y=long_form_dict["value"],
            hue=long_form_dict["control"],
            split=True,
            scale="width",
            inner="quartile",
            linewidth=0.8,
        )
        ax.set_title(f"iteration {iteration}")
        # sns.despine(left=True, bottom=True, ax=ax)

        if store_path is not None:
            plt.savefig(store_path)

        if show:
            plt.show()

        plt.close()


    def plot_sampled_actions(self, controls_dataset, iteration, show=True, store_path=None):

        episodes, timepoints, num_controls = controls_dataset.shape

        time_points = self.config.time_points
        ticks = [f"{time_point:.2f}" for time_point in time_points]

        _, axes = plt.subplots(nrows=num_controls, ncols=1, squeeze=False, sharex=True)
        axes[0][0].set_title(f"iteration {iteration}")
        axes[-1][0].set_xlabel("time")


        for time_index, time_point in enumerate(time_points):

            controls_dataset[:, time_index, :]



    #     time_points = sorted(list(action_recorder.keys()))

    #     num_controls = len(action_recorder[time_points[0]][0])
    #     controls_lists = [[[] for time in time_points] for control in range(num_controls)]

    #     for ind_time, time_point in enumerate(time_points):
    #         recorded_controls = action_recorder[time_point]

    #         for controls in recorded_controls:
    #             for ind_control, control in enumerate(controls):
    #                 controls_lists[ind_control][ind_time].append(control.item())

    #     ticks = [f"{time_point:.2f}" for time_point in time_points]

    #     _, axes = plt.subplots(nrows=num_controls, ncols=1, squeeze=False, sharex=True)

    #     axes[0][0].set_title(f"iteration {iteration}")
    #     axes[-1][0].set_xlabel("time")
    #     # plt.xticks(range(len(ticks)), ticks)
    #     for num_control, ax_row in enumerate(axes):
    #         ax = ax_row[0]
    #         sns.violinplot(
    #             data=controls_lists[num_control],
    #             ax=ax,
    #             scale="width",
    #             bw="silverman",
    #             linewidth=0.8,
    #         )
    #         sns.despine(left=True, bottom=True, ax=ax)
    #         ax.set_ylabel(f"control {num_control}")
    #         ax.set_xticklabels(ticks)

    #     if store_path is not None:
    #         plt.savefig(store_path)

    #     if show:
    #         plt.show()
    #     plt.close()


    # def plot_reward_evolution(
    #     self, all_rewards, iteration, config, show=True, store_path=None
    # ):
    #     rewards = all_rewards[:iteration]
    #     plt.plot(rewards)
    #     plt.title(
    #         f"batch size:{config.episode_batch} lr:{config.learning_rate} iteration:{iteration}"
    #     )
    #     plt.xlabel("iteration")
    #     plt.ylabel("reward")
    #     plt.xlim(-1, config.iterations + 1)
    #     mini = min(all_rewards)
    #     maxi = max(all_rewards)
    #     width = maxi - mini
    #     plt.ylim(mini - 0.1 * width, maxi + 0.1 * width)

    #     if store_path is not None:
    #         plt.savefig(store_path)
    #     if show:
    #         plt.show()
    #     plt.close()


    # def plot(self):
    #     "Plot everything from data"

    #     with h5py.File(self.filepath, mode="r") as h5file:

    #         self.dataset_iterator(h5file, self.models[0], "rewards")

    #         # prepare directories for results
    #         if not config.discard_graphics:
    #             chebyshev_labels = config.initial_controls_labels
    #             plots_dir = DIRS["figures"] / (
    #                 f"policy_{policy.__class__.__name__}_"
    #                 f"method_{config.policy_gradient_method}_"
    #                 f"batch_{config.episode_batch}_"
    #                 f"controls_{chebyshev_labels}"
    #             )
    #             plots_dir.mkdir(exist_ok=True)

    #         print(
    #             f"""
    #             Training for {self.config.iterations} iterations of
    #             {self.config.episode_batch} sampled episodes each!
    #             """
    #         )


    #             if not config.discard_graphics:

    #                 plot_name = (
    #                     f"action_distribution_"
    #                     f"model_{model.__class__.__name__}_"
    #                     f"lr_{config.learning_rate}_"
    #                     f"iteration_{iteration:03d}.png"
    #                 )
    #                 if model.controls_dims == 2:
    #                     plot_sampled_biactions(
    #                         action_recorder,
    #                         iteration,
    #                         show=False,
    #                         store_path=plots_dir / plot_name,
    #                     )
    #                 else:
    #                     plot_sampled_actions(
    #                         action_recorder,
    #                         iteration,
    #                         show=False,
    #                         store_path=plots_dir / plot_name,
    #                     )

    #         # NOTE: separated to have all rewards accesible to tune ylims accordingly
    #         if not config.discard_graphics:
    #             for iteration in range(config.iterations):
    #                 plot_name = (
    #                     f"reward_"
    #                     f"model_{model.__class__.__name__}_"
    #                     f"lr_{config.learning_rate}_"
    #                     f"iteration_{iteration:03d}.png"
    #                 )
    #                 # TODO: plot reward with std
    #                 plot_reward_evolution(
    #                     recorder["rewards_mean"],
    #                     iteration,
    #                     config,
    #                     show=False,
    #                     store_path=plots_dir / plot_name,
    #                 )
