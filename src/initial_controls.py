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


def label_cheby_identifiers(c, n):
    """
    String that represents the Chebyshev polinomial of first kind
    with the given unitary coefficient and n-th order.
    """
    assert c in {-1, 1}
    if c > 0:
        return f"T_{n}(x)"
    else:
        return f"-T_{n}(x)"


def multilabel_cheby_identifiers(list_of_ids):
    "Corresponding Chebyshev polinomial string representation for multiple (c, n) pairs."
    return "_".join([label_cheby_identifiers(c, n) for c, n in list_of_ids])


def random_chebys_generator(number, printer=True):
    "Generate random first Chebyshev polinomial functions with coefficients in {-1, 1} without replacement."
    coeffs_set = {-1, 1}
    max_order = ceildiv(number, len(coeffs_set)) + 3
    pairings = it.product(
        coeffs_set, range(1, max_order)
    )  # avoid 0 order because it falls onboundary
    sampled = random.sample(list(pairings), number)
    for c, n in sampled:
        # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        if printer:
            print("Chebyshev polinomial of first kind:", label_cheby_identifiers(c, n))
        yield (c, n), lambda x, c=c, n=n: c * cheby_basis(x, n)


def random_chebys(num_controls, time_points, zipped: bool = False):
    "Generate pretraining samples that follow Chebyshev polinomials."
    safeguard = 0.1
    id_tuples = []
    controls = []
    for id_tuple, fun in random_chebys_generator(num_controls):
        control = []
        for t in time_points:
            c = fun(t)
            c = affine_transform(c, -1, 1, 0 + safeguard, 5 - safeguard)
            control.append(c)
        id_tuples.append(id_tuple)
        controls.append(control)

    if zipped:
        return id_tuples, list(zip(*controls))
    return id_tuples, controls


if __name__ == "__main__":

    def test():
        import matplotlib.pyplot as plt

        time_points = arange(0, 1, 0.1)
        identifiers, controls = random_chebys(2, time_points)
        for i, ctrl in enumerate(controls):
            c, n = identifiers[i]
            plt.plot(time_points, ctrl, label=label_cheby_identifiers(c, n))
        plt.legend()
        plt.show()

    test()