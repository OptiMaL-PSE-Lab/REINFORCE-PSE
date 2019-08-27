"""Utility functions."""

from pathlib import Path
from numbers import Number
from itertools import zip_longest

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

DIRS = {
    "results": BASE_DIR / "results",
    "data": BASE_DIR / "results" / "data",
    "figures": BASE_DIR / "results" / "figures",
    "policies": BASE_DIR / "results" / "policies",
}

DIRS["results"].mkdir(exist_ok=True)
DIRS["data"].mkdir(exist_ok=True)
DIRS["figures"].mkdir(exist_ok=True)
DIRS["policies"].mkdir(exist_ok=True)

EPS = np.finfo(np.float32).eps.item()


def ceildiv(num, den):
    "Integer ceiled division."
    return -(-num // den)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def affine_transform(x, original_low, original_high, desired_low, desired_high):
    "Affine transform from interval [low_limit, high_limit] to [desired_low, desired_high]."

    assert (
        original_low <= x <= original_high
    ), f"Value out of bounds [{original_low}, {original_high}]."

    x = x - original_low
    x = x / (original_high - original_low)
    x = x * (desired_high - desired_low)
    x = x + desired_low
    return x


def iterable(controls):
    """Wrap control(s) in a iterable container."""
    if isinstance(controls, Number):
        return (controls,)
    return controls


def shift_grad_tracking(torch_object, track):
    """Set tracking flag for each parameter of torch object."""
    for param in torch_object.parameters():
        param.requires_grad = track
