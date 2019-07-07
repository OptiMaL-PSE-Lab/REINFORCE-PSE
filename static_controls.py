"""Here we generate a random bounded function in a closed interval."""

# NOTE: Chebyshev polinomials of the first kind are useful because they are
#       defined in the closed interval [-1,1] and bounded between [-1, 1].
#       Cheaper to evaluate and define than Fourier series.

import random
import itertools as it
from numpy import arange


def ceildiv(num, den):
    "Integer ceiled division."
    return -(-num // den)


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
    pairings = it.product(coeffs_set, range(max_order))
    sampled = random.sample(list(pairings), number)
    for c, n in sampled:
        # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        if printer:
            print(f"Selected Chebyshev polinomial of first kind: {c} * T_{n}(x)")
        yield lambda x, c=c, n=n: c * cheby_basis(x, n)


def affine_transform(x, original_low, original_high, desired_low, desired_high):
    "Affine transform from interval [low_limit, high_limit] to [desired_low, desired_high]."

    assert original_low <= x <= original_high, f"Value out of bounds [{original_low}, {original_high}]."

    x = x - original_low
    x = x / (original_high - original_low)
    x = x * (desired_high - desired_low)
    x = x + desired_low
    return x


def grouper(iterable, chunk_size):
    "Consume an iterator in exclusive chunks of given size."
    assert len(iterable) % chunk_size == 0, "Iterable length should be a multiple of chunk_size"
    ext = [iter(iterable)] * chunk_size
    return zip(*ext)


def random_chebys(num_controls, time_points, zipped: bool = False):
    "Generate pretraining samples that follow Chebyshev polinomials."
    epsilon = 0.1
    controls = []
    for fun in random_chebys_generator(num_controls):
        control = []
        for t in time_points:
            c = fun(t)
            c = affine_transform(c, -1, 1, 0 + epsilon, 5 - epsilon)
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