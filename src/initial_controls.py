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


def random_coeff_order_combinations(number):
    "Generate distinct random combinations of unitary coefficients and polynomial orders."
    coeffs_set = {-1, 1}
    max_order = ceildiv(number, len(coeffs_set)) + 3
    pairings = it.product(
        coeffs_set, range(1, max_order)
    )  # avoid 0 order because it falls onboundary
    return random.sample(list(pairings), number)


def chebys_generator(coeff_order_pairs, printer=False):
    "Generate random first Chebyshev polinomial functions with the given coefficients-order pairs."
    for c, n in coeff_order_pairs:
        # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        if printer:
            print("Chebyshev polinomial of first kind:", label_cheby_identifiers(c, n))
        yield (c, n), lambda x, c=c, n=n: c * cheby_basis(x, n)


def chebys_tracer(coef_ord_combos, time_points, zipped: bool = False):
    "Generate pretraining samples that follow Chebyshev polinomials."
    safeguard = 0.1
    coef_ord_tuples = []
    controls = []
    for coef_ord_tuple, fun in chebys_generator(coef_ord_combos):
        control = []
        for t in time_points:
            c = fun(t)
            c = affine_transform(c, -1, 1, 0 + safeguard, 5 - safeguard)
            control.append(c)
        coef_ord_tuples.append(coef_ord_tuple)
        controls.append(control)

    if zipped:
        return coef_ord_tuples, list(zip(*controls))
    return coef_ord_tuples, controls


if __name__ == "__main__":

    def test():
        import matplotlib.pyplot as plt

        time_points = arange(0, 1, 0.1)
        coef_ord_combos = random_coeff_order_combinations(2)
        identifiers, controls = chebys_tracer(coef_ord_combos, time_points)
        for i, ctrl in enumerate(controls):
            c, n = identifiers[i]
            plt.plot(time_points, ctrl, label=label_cheby_identifiers(c, n))
        plt.legend()
        plt.show()

    test()
