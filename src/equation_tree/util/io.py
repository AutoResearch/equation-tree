import json
import os
import sys
import warnings

from equation_tree.prior import default, multiply

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

PACKAGE_NAME = "equation_tree"
HASH_FILE = "_hashed_probabilities.json"


def load(prior, max_num_variables, file):
    _prior = prior.copy()
    _prior["max_num_variables"] = max_num_variables
    hash_id = str(_prior)
    default_prior = default(_prior)
    default_id = str(default_prior)
    _tmp = __load(hash_id, file)
    if _tmp is None:
        _tmp = __load_default(hash_id)
    if _tmp is None:
        warnings.warn(
            "No hashed prior found. Sample frequencies may diverge from the prior. "
            "Consider burning this prior first."
        )
        _tmp = __load(default_id, file)
        if _tmp is not None:
            return multiply(prior, _tmp)
    if _tmp is None:
        _tmp = __load_default(hash_id)
        if _tmp is not None:
            return multiply(prior, _tmp)
    if _tmp is None:
        return prior


def store(prior, max_num_variables, adjusted_prior, file):
    _prior = prior.copy()
    _prior["max_num_variables"] = max_num_variables
    hash_id = str(_prior)
    _store(hash_id, adjusted_prior, file)


def _load(hash_id, hash_fall_back, file):
    """
    Loads the hashed probabilities from the file with fall back options.
    """
    data = None
    if file:
        data = __load(hash_id, file)
    if data is None:
        data = __load_default(hash_id)
    if data is not None:
        return data

    if file:
        data = __load(hash_fall_back, file)
    if data is None:
        data = __load_default(hash_fall_back)
    if data is not None:
        warnings.warn(
            "WARNING: No hashed probabilities found for this setting. "
            "Frequencies in samples might differ from priors. "
            "Falling back to default probabilities."
        )
        return data
    warnings.warn(
        "No hashed probabilities and no default probabilities. "
        "Frequencies in samples might VASTLY differ from priors. "
        "Consider burning probabilities."
    )
    return None


def _store(hash_id, data, file):
    """
    Store hashed probabilities to a file.
    """
    if not os.path.isfile(file):
        with open(file, "w") as f:
            json.dump({hash_id: data}, f)
        return
    with open(file, "r") as f:
        _data = json.load(f)
    _data[hash_id] = data
    with open(file, "w") as f:
        json.dump(_data, f)


def __load(hash_id, file):
    if file is None:
        return None
    if not os.path.isfile(file):
        return None
    with open(file, "r") as file:
        data = json.load(file)
    if hash_id in data.keys():
        return data[hash_id]
    return None


def __load_default(hash_id):
    pkg = importlib_resources.files(PACKAGE_NAME)
    hash_file = pkg / "data" / HASH_FILE
    return __load(hash_id, hash_file)
