"""Here we generate a random bounded function in a closed interval."""

# NOTE: Chebyshev polinomials of the first kind are useful because they are
#       defined in the closed interval [-1,1] and bounded between [-1, 1].
#       Cheaper to evaluate and define than Fourier series.

import random
import itertools as it

from numpy import arange

from utils import ceildiv, affine_transform


def cheby_basis(x, n: int):
    "The n-th function of the Chebyshev polinomial basis of the first kind."
    if n == 0:
        return 1
    elif n == 1:
        return x
    elif n > 1:
        return 2 * x * cheby_basis(x, n - 1) - cheby_basis(x, n - 2)
    else:
        raise ValueError("Basis functions just defined for integer order > 0")


def random_chebys_generator(number, printer=True):
    "Generate random first Chebyshev polinomial functions with coefficients in {-1, 1} without replacement."
    coeffs_set = {-1, 1}
    max_order = ceildiv(number, len(coeffs_set)) + 3
    pairings = it.product(coeffs_set, range(1, max_order))  # avoid 0 order because it falls onboundary
    sampled = random.sample(list(pairings), number)
    for c, n in sampled:
        # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        if printer:
            print(f"Chebyshev polinomial of first kind: {c} * T_{n}(x)")
        yield lambda x, c=c, n=n: c * cheby_basis(x, n)


def random_chebys(num_controls, time_points, zipped: bool = False):
    "Generate pretraining samples that follow Chebyshev polinomials."
    safeguard = 0.1
    controls = []
    for fun in random_chebys_generator(num_controls):
        control = []
        for t in time_points:
            c = fun(t)
            c = affine_transform(c, -1, 1, 0 + safeguard, 5 - safeguard)
            control.append(c)
        controls.append(control)

    if zipped:
        return list(zip(*controls))
    return controls


if __name__ == "__main__":

    def test():
        import matplotlib.pyplot as plt
        time_points = arange(0, 1, 0.1)
        controls = random_chebys(2, time_points)
        for i, c in enumerate(controls):
            plt.plot(time_points, c, label=f"f_{i}")
        plt.legend()
        plt.show()

    test()
