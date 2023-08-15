import json
import os
import sys
import warnings

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

PACKAGE_NAME = "equation_tree"
HASH_FILE = "_hashed_probabilities.json"


def load(prior, file):
    pass

def _load(hash_id, hash_fall_back, file):
    """
    Loads the hashed probabilities from the file with fall back options.
    """
    data = None
    if file:
        data = __load(hash_id, file)
    if data is None:
        data = _load_default(hash_id)
    if data is not None:
        return data

    if file:
        data = __load(hash_fall_back, file)
    if data is None:
        data = _load_default(hash_fall_back)
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


def store(hash_id, data, file):
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
    if not os.path.isfile(file):
        return None
    with open(file, "r") as file:
        data = json.load(file)
    if hash_id in data.keys():
        return data[hash_id]
    return None


def _load_default(hash_id):
    pkg = importlib_resources.files(PACKAGE_NAME)
    hash_file = pkg / "data" / HASH_FILE
    return _load(hash_id, hash_file)
