import random
from enum import Enum
import numpy as np


class NodeType(Enum):
    NONE = 0
    FUNCTION = 1
    OPERATION = 2
    INPUT = 3
    CONSTANT = 4


class TreeNode:
    def __init__(
            self, val=0, left=None, right=None, type=NodeType.NONE, parent=None, attribute="",
            evaluation=0
    ):
        self.val = val
        self.children = []
        self.parent = parent
        self.left = left
        self.right = right
        self.attribute = attribute
        self.evaluation = evaluation
        self.type = type
        self.is_leaf = False


def _generate_parent_pointers(levels, values):
    n = len(levels)
    if n == 0:
        return None
    root = TreeNode(val=values[0])
    parent = None
    stack = [(root, levels[0])]
    for i in range(1, n):
        level = levels[i]
        node = TreeNode(val=values[i])
        while stack and level <= stack[-1][1]:
            parent = stack.pop()[0]
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
    r"""
    Iterator through regular level sequences of rooted trees.
    (only works for n >= 3)

    EXAMPLES::

        sage: from surface_dynamics.misc.plane_tree import rooted_tree_iterator
        sage: for t in rooted_tree_iterator(4): print(t)
        [0, 1, 2, 3]
        [0, 1, 2, 2]
        [0, 1, 2, 1]
        [0, 1, 1, 1]
        sage: for t in rooted_tree_iterator(5): print(t)
        [0, 1, 2, 3, 4]
        [0, 1, 2, 3, 3]
        [0, 1, 2, 3, 2]
        [0, 1, 2, 3, 1]
        [0, 1, 2, 2, 2]
        [0, 1, 2, 2, 1]
        [0, 1, 2, 1, 2]
        [0, 1, 2, 1, 1]
        [0, 1, 1, 1, 1]

        sage: for t in rooted_tree_iterator(5,verbose=True): pass
          p =    4
          prev = [0, 1, 2, 3, 4]
          save = [0, 0, 0, 0, 0]
        [0, 1, 2, 3, 4]
          p =    4
          prev = [0, 1, 2, 3, 4]
          save = [0, 0, 0, 0, 0]
        [0, 1, 2, 3, 3]
          p =    4
          prev = [0, 1, 2, 3, 4]
          save = [0, 0, 0, 0, 0]
        [0, 1, 2, 3, 2]
          p =    3
          prev = [0, 1, 2, 0, 4]
          save = [0, 0, 0, 0, 0]
        [0, 1, 2, 3, 1]
          p =    4
          prev = [0, 1, 3, 0, 4]
          save = [0, 0, 0, 2, 0]
        [0, 1, 2, 2, 2]
          p =    3
          prev = [0, 1, 2, 0, 4]
          save = [0, 0, 0, 2, 0]
        [0, 1, 2, 2, 1]
          p =    4
          prev = [0, 3, 2, 0, 4]
          save = [0, 0, 0, 1, 0]
        [0, 1, 2, 1, 2]
          p =    2
          prev = [0, 1, 0, 0, 4]
          save = [0, 0, 0, 1, 0]
        [0, 1, 2, 1, 1]
          p =    0
          prev = [0, 0, 0, 0, 4]
          save = [0, 0, 0, 1, 0]
        [0, 1, 1, 1, 1]
    """
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
    # print("PARENTS", parents_count)

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

        # parents_count represents how many children a parent has
        # for binary we want at most 2 children per parent
        if any(value > 2 for value in parents_count.values()):
            continue
        yield levels


# https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
def _bfs(s):
    # Create a queue for BFS
    queue = []

    # enqueue source node
    queue.append(s)

    while queue:
        # Dequeue a vertex from
        # queue and print it
        s = queue.pop(0)
        print("parent", s.val)

        # Get all adjacent vertices of the
        # dequeued vertex s.
        # If an adjacent has not been visited,
        # then mark it visited and enqueue it
        print("children")
        for v in s.children:
            print(v.val)
            queue.append(v)


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


def sample_tree_structure(max_depth, max_iter=1000, prior={}):
    tree_structures = _gen_all_tree_structures(max_depth)
    for _ in range(max_iter):
        idx_sample = np.random.randint(0, len(tree_structures))
        tree_structure = tree_structures[idx_sample]
        if not _is_binary_tree(tree_structure):
            print(tree_structure)
            continue
        return tree_structure
