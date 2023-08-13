import warnings
from typing import Dict

import numpy as np

from equation_tree.util.priors import set_priors

MAX_ITER = 1000


class _StructureNode:
    def __init__(self, val=0):
        self.val = val
        self.parent = None
        self.children = []


def _generate_parent_pointers(levels, values):
    n = len(levels)
    if n == 0:
        return None
    root = _StructureNode(val=values[0])
    stack = [(root, levels[0])]
    for i in range(1, n):
        level = levels[i]
        node = _StructureNode(val=values[i])
        while stack and level <= stack[-1][1]:
            stack.pop()[0]
        parent = stack[-1][0]

        if parent:
            parent.children.append(node)
            node.parent = parent

        stack.append((node, level))
    return root


def _dfs_parents(node, parents_count):
    if not node:
        return

    if node.parent is not None:
        parents_count[node.parent.val] += 1

    for child in node.children:
        _dfs_parents(child, parents_count)


def _is_binary_tree(tree_structure):
    parents_count = [0] * len(tree_structure)
    root = _generate_parent_pointers(tree_structure, range(len(tree_structure)))
    _dfs_parents(root, parents_count)
    return all(count <= 2 for count in parents_count)


def _rooted_tree_iterator(n, verbose=False):
    assert n >= 3

    levels = list(range(n))
    prev = list(range(n))  # function: level -> ?
    save = [0] * n
    p = n - 1

    if verbose:
        print("  p =    %s" % p)
        print("  prev = %s" % prev)
        print("  save = %s" % save)
        print(levels)
    root = _generate_parent_pointers(levels, np.arange(n))
    parents_count = {i: 0 for i in range(n)}
    _dfs_parents(root, parents_count)

    yield levels

    while p > 0:
        levels[p] = levels[p] - 1
        if p < n and (levels[p] != 1 or levels[p - 1] != 1):
            diff = p - prev[levels[p]]  # = p-q
            while p < n - 1:
                save[p] = prev[levels[p]]
                prev[levels[p]] = p
                p += 1
                levels[p] = levels[p - diff]
        while levels[p] == 1:
            p -= 1
            prev[levels[p]] = save[p]

        if verbose:
            print("  p =    %s" % p)
            print("  prev = %s" % prev)
            print("  save = %s" % save)
            print(levels)
        root = _generate_parent_pointers(levels, np.arange(n))
        parents_count = {i: 0 for i in range(n)}
        _dfs_parents(root, parents_count)

        if any(value > 2 for value in parents_count.values()):
            continue
        yield levels


def _get_children(tree_structure, index):
    parent = tree_structure[index]
    children = []
    for i in range(index + 1, len(tree_structure)):
        if tree_structure[i] == parent:
            break
        if tree_structure[i] == parent + 1:
            children.append(i)
    return children


def _count_children(tree_structure, index):
    parent = tree_structure[index]
    children = 0
    for i in range(index + 1, len(tree_structure)):
        if tree_structure[i] == parent:
            break
        if tree_structure[i] == parent + 1:
            children += 1
    return children


def _gen_all_tree_structures(max_depth):
    tree_structures = [
        tree.copy()
        for depth in range(3, max_depth + 1)
        for tree in _rooted_tree_iterator(depth)
    ]
    return tree_structures


def sample_tree_structure(max_depth: int, priors: Dict = {}):
    """
    Sample a tree structure.

    Args:
        max_depth: the maximum depth of a tree
        priors: priors in form of a dictionary. The keys are the tree structures as strings

    Examples:
        # Set seed for reproducibility
        >>> np.random.seed(42)

        # Sample a single structure
        >>> sample_tree_structure(7)
        [0, 1, 2, 3, 3, 2]

        # Get a list of 4 samples with max_depth = 4
        >>> [sample_tree_structure(5) for _ in range(4)]
        [[0, 1, 2, 1, 2], [0, 1, 2, 3, 1], [0, 1, 2, 3, 3], [0, 1, 1]]

        # Set a prior
        >>> [sample_tree_structure(5, {'[0, 1, 1]': 1}) for _ in range(4)]
        [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]

        # Set a different prior
        >>> [sample_tree_structure(5, {'[0, 1, 1]': .5, '[0, 1, 2]': .5}) for _ in range(4)]
        [[0, 1, 1], [0, 1, 2], [0, 1, 1], [0, 1, 1]]

        # We can also set a full prior( here the max_depth for the structure is 3 which only allows
        # for two structures: [0, 1, 1] and [0, 1, 2]
        >>> [sample_tree_structure(3, {'[0, 1, 1]': .3, '[0, 1, 2]': .7}) for _ in range(4)]
        [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]

        # If we set an invalid prior, an Exeption is raised  # doctest: +ELLIPSIS
        >>> [sample_tree_structure(3, {'[0, 1, 3]': .5}) for _ in range(4)]
        Traceback (most recent call last):
        ...
        Exception: Priors {'[0, 1, 3]': 0.5} are not subset of space ['[0, 1, 2]', '[0, 1, 1]']

    """
    tree_structures = _gen_all_tree_structures(max_depth)

    priors = set_priors(priors, [str(structure) for structure in tree_structures])
    probabilities = [priors[key] for key in priors.keys()]

    for _ in range(MAX_ITER):
        sample_index = np.random.choice(len(tree_structures), p=probabilities)
        tree_structure = tree_structures[sample_index]
        if not _is_binary_tree(tree_structure):
            warnings.warn(
                "Found a non binary tree structure, "
                "this might lead to discrepancies between set priors and sample frequencies."
            )
            continue
        return tree_structure
    raise Exception(f"Could not generate tree structure with max depth {max_depth}")
