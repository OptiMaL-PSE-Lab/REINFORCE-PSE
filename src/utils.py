"""Utility functions."""

from pathlib import Path
from numbers import Number

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
POLICIES_DIR = BASE_DIR / "policies"

FIGURES_DIR.mkdir(exist_ok=True)
POLICIES_DIR.mkdir(exist_ok=True)

EPS = np.finfo(np.float32).eps.item()


def ceildiv(num, den):
    "Integer ceiled division."
    return -(-num // den)


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
