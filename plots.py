
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.ticker import FuncFormatter

# sns.set(style="whitegrid")

def plot_episode_states(time_array, y1, y2, U, objective=None, fontsize=12,
                        show=True, store_path=None):

    grid_shape = (3,1)

    plt.subplot2grid(grid_shape, (0, 0))
    plt.plot(time_array, y1)
    plt.grid(axis='x') # , ls='--', lw=.5, c='k', alpha=.3
    plt.ylabel('y1', fontsize=fontsize)
    plt.xlabel('time', fontsize=fontsize)

    plt.subplot2grid(grid_shape, (1, 0))
    plt.plot(time_array, y2)
    plt.grid(axis='x')
    plt.ylabel('y2', fontsize=fontsize)
    plt.xlabel('time', fontsize=fontsize)

    plt.subplot2grid(grid_shape, (2, 0))
    plt.plot(time_array, U)
    if objective is not None:
        plt.plot(time_array, objective, 'ro')
    plt.grid(axis='x')
    plt.ylabel('u', fontsize=fontsize)
    plt.xlabel('time', fontsize=fontsize)
    plt.ylim(0, 5)

    if store_path is not None:
        plt.savefig(store_path)
    if show:
        plt.show()
    plt.close()

def plot_sampled_actions(action_recorder, iteration, show=True, store_path=None):

    dataframe = pd.DataFrame(action_recorder) # NOTE: avoid pandas in the future...

    f = plt.figure(figsize=(15, 6))
    with sns.axes_style("whitegrid"):
        sns.violinplot(data=dataframe)
        sns.despine(left=True, bottom=True)
        plt.xlabel('time')
        plt.ylabel('action')
        plt.title(f"iteration {iteration}")
        # plt.gca().xaxis.set_major_formatter(FuncFormatter("{:.2f}".format)) # FIXME

    if store_path is not None:
        plt.savefig(store_path, dpi=800)
    if show:
        plt.show()
    plt.close()
