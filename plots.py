from os.path import join

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rc("figure", figsize=(9, 6))
mpl.rc("savefig", bbox="tight", dpi=500)


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


def plot_sampled_actions(action_recorder, iteration, show=True, store_path=None):

    time_points = sorted(list(action_recorder.keys()))
    arrays = [action_recorder[time_point] for time_point in time_points]
    ticks = [f"{time_point:.2f}" for time_point in time_points]

    plt.figure(figsize=(12, 6))
    with sns.axes_style("whitegrid"):
        sns.violinplot(data=arrays)
        sns.despine(left=True, bottom=True)
        plt.xlabel("time")
        plt.ylabel("action")
        plt.title(f"iteration {iteration}")
        plt.xticks(range(len(ticks)), ticks)

    if store_path is not None:
        plt.savefig(store_path)
    if show:
        plt.show()
    plt.close()


def plot_reward_evolution(rewards, iteration, opt_specs, show=True, store_path=None):
    plt.plot(rewards)
    plt.title(
        f"batch size:{opt_specs['episode_batch']} lr:{opt_specs['learning_rate']} iteration:{iteration}"
    )
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.xlim(0, opt_specs["iterations"])

    if store_path is not None:
        plt.savefig(store_path)
    if show:
        plt.show()
    plt.close()
