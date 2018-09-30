from matplotlib import pyplot as plt

def plot_state_policy_evol(time_array, y1, y2, U, objective=None, fontsize=12,
                           show=True, store_path=None):

    grid_shape = (3,1)
    # fig = plt.figure()

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

    # fig.suptitle("test parameters")

    if store_path is not None:
        plt.savefig(store_path)
        plt.close()
    if show:
        plt.show()