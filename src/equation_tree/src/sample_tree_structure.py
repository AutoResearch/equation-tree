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


def _gen_all_tree_structures_full(max_depth):
    """
    Examples:
        >>> trees = _gen_all_tree_structures_full(3)
        >>> [_get_list(t) for t in trees]

    """
    if max_depth == 0:
        return [None]

    result = []
    for left_count in range(max_depth):
        right_count = max_depth - 1 - left_count
        left_trees = _gen_all_tree_structures_full(left_count)
        right_trees = _gen_all_tree_structures_full(right_count)

        for left_tree in left_trees:
            for right_tree in right_trees:
                root = _StructureNode()
                root.children.append(left_tree)
                root.children.append(right_tree)
                result.append(root)

    return result


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


def convert_to_standard_notation(list_notation):
    """
    Examples:
        >>> convert_to_standard_notation([0, 1, 1])
        [0, 1, 1]
        >>> convert_to_standard_notation([0, 1, 2, 2, 3])
        [0, 1, 2, 3, 2]
        >>> convert_to_standard_notation([0, 1, 2, 1, 2, 2])
        [0, 1, 2, 2, 1, 2]
        >>> convert_to_standard_notation([0, 1, 2, 2, 1, 2, 3])
        [0, 1, 2, 3, 1, 2, 2]
        >>> convert_to_standard_notation([0, 1, 2, 3, 1, 2, 2, 3])
        [0, 1, 2, 3, 2, 1, 2, 3]
    """
    tree = _get_tree(list_notation)
    _standardize(tree)
    return _get_list(tree)


def _get_tree(list_notation):
    """
    Example:
        >>> t = _get_tree([0])
        >>> t.val
        0
        >>> t.parent

        >>> t.children
        []
        >>> t = _get_tree([0, 1, 1])
        >>> t.val
        0
        >>> t.parent

        >>> t.children[0].val
        1
        >>> t.children[0].parent.val
        0
        >>> t.children[0].children
        []
        >>> t.children[1].val
        1
        >>> t.children[1].parent.val
        0
        >>> t.children[1].children
        []

        >>> t = _get_tree([0, 1, 2, 3, 2, 3])
        >>> _get_depth(t)
        3
        >>> _get_width(t)
        2

        >>> t = _get_tree([0, 1, 1, 2, 2, 3])
        >>> _get_depth(t)
        3
        >>> _get_width(t)
        3

    """
    node = _StructureNode(0)
    root = node
    last_val = 0
    for val in list_notation[1:]:
        if val > last_val:
            last_node = node
            node.children.append(_StructureNode(val))
            node = node.children[-1]
            node.parent = last_node
        if val <= last_val:
            tmp = val
            while tmp <= last_val:
                node = node.parent
                tmp += 1
            last_node = node
            node.children.append(_StructureNode(val))
            node = node.children[-1]
            node.parent = last_node
        last_val = val
    return root


def _get_list(node):
    lst = []

    def _rec_list(n):
        nonlocal lst
        lst.append(n.val)
        for c in n.children:
            _rec_list(c)

    _rec_list(node)
    return lst


def _standardize(node):
    def _rec_standardize(n):
        if len(n.children) > 1:
            if _get_width(n.children[1]) > _get_width(n.children[0]):
                n.children[0], n.children[1] = n.children[1], n.children[0]
            if _get_depth(n.children[1]) > _get_depth(n.children[0]):
                n.children[0], n.children[1] = n.children[1], n.children[0]
        if n.children:
            _rec_standardize(n.children[0])
        if len(n.children) > 1:
            _rec_standardize(n.children[1])

    _rec_standardize(node)


def _get_depth(node):
    depth = 0

    def _rec_depth(n):
        nonlocal depth
        if n.children:
            if n.children[0].val > depth:
                depth = n.children[0].val
            _rec_depth(n.children[0])
            if len(n.children) > 1:
                if n.children[1].val > depth:
                    depth = n.children[1].val
                _rec_depth(n.children[1])

    _rec_depth(node)
    return depth


def _get_width(node):
    width = 0

    def _rec_width(n):
        nonlocal width
        if n.children:
            _rec_width(n.children[0])
        if len(n.children) > 1:
            _rec_width(n.children[1])
        if not n.children:
            width += 1

    _rec_width(node)
    return width
