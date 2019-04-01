from os.path import join

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

sns.set_style(style="whitegrid")

mpl.rc('font', size=SMALL_SIZE)        # controls default text sizes
mpl.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

mpl.rc("figure", figsize=(6, 2))
mpl.rc("savefig", bbox="tight", dpi=600)


def plot_episode_states(
    time_array, y1, y2, U, objective=None, fontsize=12, show=True, store_path=None
):
    grid_shape = (3, 1)

    plt.subplot2grid(grid_shape, (0, 0))
    plt.plot(time_array, y1)
    plt.grid(axis="x")  # , ls='--', lw=.5, c='k', alpha=.3
    plt.ylabel("y1", fontsize=fontsize)
    plt.xlabel("time", fontsize=fontsize)

    plt.subplot2grid(grid_shape, (1, 0))
    plt.plot(time_array, y2)
    plt.grid(axis="x")
    plt.ylabel("y2", fontsize=fontsize)
    plt.xlabel("time", fontsize=fontsize)

    plt.subplot2grid(grid_shape, (2, 0))
    plt.plot(time_array, U)
    if objective is not None:
        plt.plot(time_array, objective, "ro")
    plt.grid(axis="x")
    plt.ylabel("u", fontsize=fontsize)
    plt.xlabel("time", fontsize=fontsize)
    plt.ylim(0, 5)

    if store_path is not None:
        plt.savefig(store_path)
    if show:
        plt.show()
    plt.close()


def plot_sampled_biactions(action_recorder, iteration, show=True, store_path=None):

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


def plot_sampled_actions(action_recorder, iteration, show=True, store_path=None):

    time_points = sorted(list(action_recorder.keys()))

    num_controls = len(action_recorder[time_points[0]][0])
    controls_lists = [[[] for time in time_points] for control in range(num_controls)]

    for ind_time, time_point in enumerate(time_points):
        recorded_controls = action_recorder[time_point]

        for controls in recorded_controls:
            for ind_control, control in enumerate(controls):
                controls_lists[ind_control][ind_time].append(control.item())

    ticks = [f"{time_point:.2f}" for time_point in time_points]

    _, axes = plt.subplots(nrows=num_controls, ncols=1, squeeze=False, sharex=True)

    axes[0][0].set_title(f"iteration {iteration}")
    axes[-1][0].set_xlabel("time")
    # plt.xticks(range(len(ticks)), ticks)
    for num_control, ax_row in enumerate(axes):
        ax = ax_row[0]
        sns.violinplot(
            data=controls_lists[num_control],
            ax=ax,
            scale="width",
            bw="silverman",
            linewidth=0.8,
        )
        sns.despine(left=True, bottom=True, ax=ax)
        ax.set_ylabel(f"control {num_control}")
        ax.set_xticklabels(ticks)

    if store_path is not None:
        plt.savefig(store_path)

    if show:
        plt.show()
    plt.close()


def plot_reward_evolution(
    all_rewards, iteration, opt_specs, show=True, store_path=None
):
    rewards = all_rewards[:iteration]
    plt.plot(rewards)
    plt.title(
        f"batch size:{opt_specs['episode_batch']} lr:{opt_specs['learning_rate']} iteration:{iteration}"
    )
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.xlim(-1, opt_specs["iterations"] + 1)
    mini = min(all_rewards)
    maxi = max(all_rewards)
    width = maxi - mini
    plt.ylim(mini - 0.1 * width, maxi + 0.1 * width)

    if store_path is not None:
        plt.savefig(store_path)
    if show:
        plt.show()
    plt.close()
