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

    def __init__(self, model_name, config):

        self.model_name = model_name
        self.config = config

        with h5py.File(config.data_file, mode="r") as h5file:
            self.seeds = h5file.attrs["seeds"]
            self.models = h5file.attrs["models"]
            self.time_points = h5file.attrs["time_points"]

            model_group = h5file[f"model_{self.model_name}"]
            self.iterations = model_group.attrs["iterations"]
            self.learning_rate = model_group.attrs["learning_rate"]

    def samples_stack(self, iteration, variable):
        "Retrieve and stack datasets corresponding to distinct random seeds."

        assert variable in {"states", "controls", "rewards"}

        def path_forger(seed):
            return f"model_{self.model_name}/iter_{iteration}/{variable}/seed_{seed}"

        with h5py.File(self.config.data_file, mode="r") as h5file:
            # pylint: disable=not-an-iterable  # pylint shortcoming
            datasets = [h5file[path_forger(seed)][:] for seed in self.seeds]

        return np.stack(datasets, axis=0)  # stack over new dimension as first new index

    # TODO: handle this n-controls case
    def plot_sampled_biactions(self):

        figures_dir = self.config.figures_dir / f"model_{self.model_name}" / "actions"
        figures_dir.mkdir(parents=True)

        for iteration in range(self.iterations):

            # dataset_shape : (seeds, rool_outs, time, control_dim)
            controls_dataset = self.samples_stack(iteration, "controls")
            num_controls = controls_dataset.shape[-1]

            assert (
                num_controls == 2
            ), "Plotting routine is just intended for 2 controls."

            # seaborn violinplots require all data in 1-dimensional arrays
            long_form_dict = {key: [] for key in ["time", "value", "control"]}
            for t_ind, time_point in enumerate(self.time_points):
                for n_control in range(num_controls):

                    samples = controls_dataset[:, :, t_ind, n_control].reshape(-1)
                    num_samples = len(samples)

                    long_form_dict["value"] = np.concatenate(
                        (long_form_dict["value"], samples)
                    )
                    long_form_dict["time"] = np.concatenate(
                        (long_form_dict["time"], [f"{time_point:.2f}"] * num_samples)
                    )
                    long_form_dict["control"] = np.concatenate(
                        (
                            long_form_dict["control"],
                            [f"control_{n_control}"] * num_samples,
                        )
                    )

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

            plt.savefig(figures_dir / f"iter_{iteration}")
            # plt.show()
            plt.close()

    def plot_state_evolution(self):

        figures_dir = self.config.figures_dir / f"model_{self.model_name}" / "states"
        figures_dir.mkdir(parents=True)

        for iteration in range(self.iterations):

            # dataset_shape : (seeds, rool_outs, time, control_dim)
            states_dataset = self.samples_stack(iteration, "states")
            num_states = states_dataset.shape[-1]
            num_timepoints = states_dataset.shape[-2]

            states_mean = np.empty((num_timepoints, num_states))
            states_deviation = np.empty((num_timepoints, num_states))

            for state in range(num_states):
                for t_ind, _ in enumerate(self.time_points):
                    samples = states_dataset[:, :, t_ind, state]
                    states_mean[t_ind, state] = np.mean(samples)
                    states_deviation[t_ind, state] = np.std(samples)

            upper_bounds = states_mean + states_deviation
            lower_bounds = states_mean - states_deviation

            mini = np.min(lower_bounds)
            maxi = np.max(upper_bounds)
            width = maxi - mini

            for state in range(num_states):
                plt.plot(
                    self.time_points, states_mean[:, state], label=f"variable_{state}"
                )
                plt.fill_between(
                    self.time_points,
                    lower_bounds[:, state],
                    upper_bounds[:, state],
                    alpha=0.4,
                )
            plt.title(
                f"batch size:{self.config.episode_batch} lr:{self.learning_rate} iteration:{iteration}"
            )
            plt.xlabel("time")
            plt.ylabel("state")
            # plt.xlim(min(self.time_points), min(self.time_points))
            plt.ylim(mini - 0.1 * width, maxi + 0.1 * width)
            plt.legend()

            plt.savefig(figures_dir / f"iter_{iteration}")
            # plt.show()
            plt.close()

    def plot_reward_evolution(self):

        figures_dir = self.config.figures_dir / f"model_{self.model_name}" / "rewards"
        figures_dir.mkdir(parents=True)

        # dataset_shape : (seeds, rool_outs)
        rewards = np.empty(self.iterations)
        deviations = np.empty(self.iterations)
        for iteration in range(self.iterations):
            rewards_dataset = self.samples_stack(iteration, "rewards")
            rewards[iteration] = np.mean(rewards_dataset)
            deviations[iteration] = np.std(rewards_dataset)

        upper_bound = rewards + deviations
        lower_bound = rewards - deviations

        mini = min(lower_bound)
        maxi = max(upper_bound)
        width = maxi - mini

        for iteration in range(2, self.iterations):
            plt.plot(rewards[: iteration + 1])
            plt.fill_between(
                range(iteration + 1),
                lower_bound[: iteration + 1],
                upper_bound[: iteration + 1],
                alpha=0.4,
            )
            plt.title(
                f"batch size:{self.config.episode_batch} lr:{self.learning_rate} iteration:{iteration}"
            )
            plt.xlabel("iteration")
            plt.ylabel("reward")
            plt.xlim(-1, self.iterations + 1)
            plt.ylim(mini - 0.1 * width, maxi + 0.1 * width)

            plt.savefig(figures_dir / f"iter_{iteration}")
            # plt.show()
            plt.close()

    def plot_everything(self):
        self.plot_sampled_biactions()
        self.plot_reward_evolution()
        self.plot_state_evolution()

