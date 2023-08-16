"""
Do not run this in a package. This file is used to burn
"""
import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from equation_tree.util.io import HASH_FILE, PACKAGE_NAME, load, store


def hash_load(hash_id):
    return load(hash_id)


def hash_store(hash_id, data):
    pkg = importlib_resources.files(PACKAGE_NAME)
    hash_file = pkg / "data" / HASH_FILE
    store(hash_id, data, hash_file)


hash_store("ab", {"c": 1})
