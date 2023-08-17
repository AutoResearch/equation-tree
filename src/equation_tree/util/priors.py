import warnings
from typing import List


def priors_from_space(space: List):
    _lst = list(set(space))
    res = {}
    for key in _lst:
        res[key] = 1 / len(_lst)
    return res


def set_priors(priors=None, space=None):
    """
    Utility function to set priors without setting all probabilities of the space

    Examples:
        >>> default_priors = set_priors(space=['a', 'b', 'c', 'd'])
        >>> default_priors
        {'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25}

        >>> custom_priors_full = set_priors({'a' : .3, 'b': .7}, ['a', 'b'])
        >>> custom_priors_full
        {'a': 0.3, 'b': 0.7}

        >>> custom_priors_partial = set_priors({'a' : .5}, ['a', 'b', 'c'])
        >>> custom_priors_partial
        {'a': 0.5, 'b': 0.25, 'c': 0.25}
    """
    if space is None:
        space = []
    if priors is None:
        priors = {}

    n = len(space)
    default_prior = 1 / n

    # Set all to default to begin with
    _priors = {el: default_prior for el in space}

    # If the user provides priors
    if priors:
        if not set(priors.keys()).issubset(set(space)):
            raise Exception(f"Priors {priors} are not subset of space {space}")
        total_custom_prior = sum(priors.values())
        if total_custom_prior > 1:
            raise ValueError(f"Sum of custom priors {priors} is greater than 1")

        # Update the priors dict with custom values
        for key, value in priors.items():
            _priors[key] = value

        # Adjust the other priors
        remaining_probability = 1 - total_custom_prior
        num_unset_possibilities = n - len(priors)
        for key in _priors:
            if key not in priors:
                _priors[key] = remaining_probability / num_unset_possibilities
    return _normalize(_priors)


def _normalize(priors):
    """normalize priors"""
    total = sum(priors.values())
    if total <= 0:
        warnings.warn(
            f"Sum of priors {priors} is less then 0. Falling back to default priors."
        )
        n = len(priors.keys())
        default_prior = 1 / n
        return {el: default_prior for el in priors.keys()}
    return {el: priors[el] / total for el in priors.keys()}


def normalized_dict(d):
    """
    Examples:
        >>> t_d = {'a': 1, 'b': 3}
        >>> t_d = normalized_dict(t_d)
        >>> t_d
        {'a': 0.25, 'b': 0.75}
        >>> t_d = {'c': 0, 'd': 0}
        >>> t_d = normalized_dict(t_d)
        >>> t_d
        {'c': 0, 'd': 0}
    """
    n = sum(d.values())
    if n == 0:
        return d
    else:
        return {key: value / n for key, value in d.items()}
