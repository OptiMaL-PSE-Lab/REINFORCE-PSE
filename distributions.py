import numpy as np

from torch.distributions import Beta, TransformedDistribution
from torch.distributions.transforms import AffineTransform

eps = np.finfo(np.float32).eps.item()


def forge_distribution(mean, sigma, lower_limit=0.0, upper_limit=5.0):
    """
    Find the required concentration hyperparameters in the canonical Beta distribution
    that will return the desired mean and deviation after the affine transformation.
    """
    width = upper_limit - lower_limit
    assert width > 0
    assert sigma < eps + width / 2, f"invalid std: {sigma.item()}"

    canonical_mean = (mean - lower_limit) / width
    canonical_sigma = sigma / width ** 2

    alpha_plus_beta = (canonical_mean * (1 - canonical_mean) / canonical_sigma ** 2) - 1
    alpha = canonical_mean * alpha_plus_beta
    beta = (1 - canonical_mean) * alpha_plus_beta

    canonical = Beta(alpha, beta)
    transformation = AffineTransform(loc=lower_limit, scale=width)
    transformed = TransformedDistribution(canonical, transformation)

    return transformed


def sample_actions(means, sigmas):
    """
    Forge a distribution for each pair of means and sigmas,
    sample an action from it and calculate its probability logarithm
    """

    actions = []
    sum_log_prob = 0.0
    for mean, sigma in zip(means, sigmas):

        dist = forge_distribution(mean, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        actions.append(action)
        sum_log_prob = sum_log_prob + log_prob

    return actions, sum_log_prob


def get_log_prob(means, sigmas, controls):
    """
    Forge the corresponding distributions for the given means and sigmas
    and calculate the log probability of the given controls for thos distributions.
    """
    sum_log_prob = 0.0
    for ind, (mean, sigma) in enumerate(zip(means, sigmas)):

        dist = forge_distribution(mean, sigma)
        log_prob = dist.log_prob(controls[ind])

        sum_log_prob = sum_log_prob + log_prob

    return sum_log_prob
