from enum import Enum
from typing import Callable, Dict, List

import numpy as np

from equation_tree.src.sample_tree_structure import (
    _count_children,
    _get_children,
    sample_tree_structure,
    sample_tree_structure_fast,
)
from equation_tree.util.priors import set_priors
from equation_tree.util.type_check import is_known_constant, is_numeric

MAX_ITER = 1000


class NodeKind(Enum):
    NONE = 0
    FUNCTION = 1
    OPERATOR = 2
    VARIABLE = 3
    CONSTANT = 4


class TreeNode:
    def __init__(
        self,
        left=None,
        right=None,
        kind=NodeKind.NONE,
        attribute="",
    ):
        """
        Examples:
            >>> gc_1 = TreeNode(attribute='grand_child_1')
            >>> gc_1.size
            1
            >>> child_1 = TreeNode(attribute='child_1', left=gc_1)
            >>> child_1.size
            2
            >>> child_2 = TreeNode(attribute='child_2')
            >>> child_2.size
            1
            >>> root = TreeNode(left=child_1,right=child_2,attribute='root')
            >>> root.size
            4
        """

        self.attribute = attribute
        self.kind = kind
        self.parent = None
        self.left = left
        self.right = right
        self.evaluation = 0

        self.children = []
        if left is not None:
            self.children.append(left)
            left.parent = self
        if right is not None:
            self.children.append(right)
            right.parent = self
        self.is_leaf = self.children == []

    def check_validity(
        self,
        zero_representations=["0"],
        log_representations=["log", "Log"],
        division_representations=["/", ":"],
        verbose=False,
    ):
        return check_node_validity(
            self,
            zero_representations,
            log_representations,
            division_representations,
            verbose,
        )

    @property
    def size(self):
        s_ = 1
        for c in self.children:
            s_ += c.size
        return s_


class _HelperTree:
    def __init__(self, root):
        self.root = root
        self.nodes = []
        self.ids = []
        self.lmds = []
        self.keyroots = None

        # Initialize stacks for traversal
        stack = [(root, [])]
        pstack = []

        j = 0  # Node counter

        while stack:
            node, ancestors = stack.pop()
            node_id = j

            # Traverse child nodes
            for child in node.children:
                child_ancestors = ancestors.copy()
                child_ancestors.insert(0, node_id)  # Prepend the current node_id
                stack.append((child, child_ancestors))

            pstack.append(((node, node_id), ancestors))
            j += 1

        lmds = {}
        keyroots = {}
        i = 0

        while pstack:
            (node, node_id), ancestors = pstack.pop()
            self.nodes.append(node)
            self.ids.append(node_id)

            if not node.children:
                lmd = i
                for ancestor_id in ancestors:
                    if ancestor_id not in lmds:
                        lmds[ancestor_id] = i
                    else:
                        break
            else:
                lmd = lmds.get(node_id, None)

            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1

        self.keyroots = sorted(keyroots.values())


def edit_distance(tree_a: TreeNode, tree_b: TreeNode):
    """
    Computes the tree edit distance between trees A and B

    Implements the algorithm described here:
        `Zhang, K., & Shasha, D. (1989).
        Simple fast algorithms for the editing distance between trees and related problems.
        SIAM journal on computing, 18(6), 1245-1262.`
    Inspired and adjusted(simplified) for the purpose of this repository from:
        `https://github.com/timtadh/zhang-shasha`

    Args:
        tree_a: Root of tree A
        tree_b: Root of tree B

    Return:
        An integer distance [0, inf+)
    Examples:
        >>> gc_1_a = TreeNode(attribute='grand_child_1')
        >>> gc_1_b = TreeNode(attribute='grand_child_1')
        >>> edit_distance(gc_1_a, gc_1_b)
        0
        >>> child_1_a = TreeNode(attribute='child_1', left=gc_1_a)
        >>> child_1_b = TreeNode(attribute='child_1_', left=gc_1_b)
        >>> edit_distance(child_1_a, child_1_b)
        1
        >>> child_2_a = TreeNode(attribute='child_2')
        >>> child_2_b = TreeNode(attribute='child_2')
        >>> edit_distance(child_2_a, child_2_b)
        0
        >>> root_a = TreeNode(attribute='root', left=child_1_a, right=child_2_a)
        >>> root_b = TreeNode(attribute='root', left=child_1_b, right=child_2_b)
        >>> edit_distance(root_a, root_b)
        1
        >>> root_a_ = TreeNode(attribute='root', left=child_2_a, right=child_1_a)
        >>> edit_distance(root_a, root_a_)
        2
        >>> edit_distance(root_a_, root_b)
        3

    """
    tree_a_helper = _HelperTree(tree_a)
    tree_b_helper = _HelperTree(tree_b)

    treedists = np.zeros((tree_a.size, tree_b.size), float)

    def update_cost(node_a, node_b):
        return int(node_a.attribute != node_b.attribute)

    def treedist(i, j):
        al = tree_a_helper.lmds
        bl = tree_b_helper.lmds

        an = tree_a_helper.nodes
        bn = tree_b_helper.nodes

        m = i - al[i] + 2
        n = j - bl[j] + 2

        ioff = al[i] - 1
        joff = bl[j] - 1

        # Initialize the fd array
        fd = np.zeros((m, n), float)
        fd[1:, 0] = np.arange(1, m)
        fd[0, 1:] = np.arange(1, n)

        partial_ops = [[([1] if x == 0 else []) for x in range(n)] for _ in range(m)]

        for x in range(1, m):
            for y in range(1, n):
                if al[i] == al[x + ioff] and bl[j] == bl[y + joff]:
                    costs = [
                        fd[x - 1][y] + 1,
                        fd[x][y - 1] + 1,
                        fd[x - 1][y - 1] + update_cost(an[x + ioff], bn[y + joff]),
                    ]
                    fd[x][y] = min(costs)
                    min_index = costs.index(fd[x][y])
                    if min_index == 0:
                        partial_ops[x][y] = partial_ops[x - 1][y] + [1]
                    elif min_index == 1:
                        partial_ops[x][y] = partial_ops[x][y - 1] + [1]
                    else:
                        partial_ops[x][y] = partial_ops[x - 1][y - 1] + [1]

                    treedists[x + ioff][y + joff] = fd[x][y]
                else:
                    p = al[x + ioff] - 1 - ioff
                    q = bl[y + joff] - 1 - joff
                    costs = [
                        fd[x - 1][y] + 1,
                        fd[x][y - 1] + 1,
                        fd[p][q] + treedists[x + ioff][y + joff],
                    ]
                    fd[x][y] = min(costs)
                    min_index = costs.index(fd[x][y])
                    if min_index == 0:
                        partial_ops[x][y] = partial_ops[x - 1][y] + [1]
                    elif min_index == 1:
                        partial_ops[x][y] = partial_ops[x][y - 1] + [1]
                    else:
                        partial_ops[x][y] = partial_ops[p][q] + [1]

    for i in tree_a_helper.keyroots:
        for j in tree_b_helper.keyroots:
            treedist(i, j)
    return int(treedists[-1][-1])


def ned(tree_a: TreeNode, tree_b: TreeNode):
    """
    Normalized tree edit distance according to:
    `Li, Y., & Chenguang, Z. (2011). A metric normalization of tree edit distance.
    Frontiers of Computer Science in China, 5, 119-125.`
    Examples:
        >>> gc_1_a = TreeNode(attribute='grand_child_1')
        >>> gc_1_b = TreeNode(attribute='grand_child_1')
        >>> ned(gc_1_a, gc_1_b)
        0.0
        >>> child_1_a = TreeNode(attribute='child_1', left=gc_1_a)
        >>> child_1_b = TreeNode(attribute='child_1_', left=gc_1_b)
        >>> ned(child_1_a, child_1_b)
        0.4
        >>> child_2_a = TreeNode(attribute='child_2')
        >>> child_2_b = TreeNode(attribute='child_2')
        >>> ned(child_2_a, child_2_b)
        0.0
        >>> root_a = TreeNode(attribute='root', left=child_1_a, right=child_2_a)
        >>> root_b = TreeNode(attribute='root', left=child_1_b, right=child_2_b)
        >>> ned(root_a, root_b)
        0.2222222222222222
        >>> root_a_ = TreeNode(attribute='root', left=child_2_a, right=child_1_a)
        >>> ned(root_a, root_a_)
        0.4
        >>> ned(root_a_, root_b)
        0.5454545454545454
    """
    ed = edit_distance(tree_a, tree_b)
    return 2 * ed / (tree_a.size + tree_b.size + ed)


def check_node_validity(
    node=None,
    zero_representations=["0"],
    log_representations=["log", "Log"],
    division_representations=["/", ":"],
    verbose=False,
):
    if node is None:
        return True

    if node.kind == NodeKind.FUNCTION:
        if node.left is None or node.right is not None:
            return False
        if (
            node.attribute in log_representations
            and node.left.attribute in zero_representations
        ):
            if verbose:
                print("logarithm is applied to 0 which is results in not real number.")
            return False
        if node.left.kind == NodeKind.CONSTANT:  # unnecessary complexity
            if verbose:
                print(
                    f"{node.left.attribute} is a constant "
                    f"applied to a function {node.attribute}"
                )
            return False
        return check_node_validity(node.left, verbose=verbose)

    elif node.kind == NodeKind.OPERATOR:
        if node.left is None or node.right is None:
            return False
        if (
            node.attribute in division_representations
            and node.right.attribute in zero_representations
        ):
            if verbose:
                print("division by 0 is not allowed.")
            return False
        if node.left.kind == NodeKind.CONSTANT and node.right.kind == NodeKind.CONSTANT:
            if verbose:
                print(
                    f"{node.left.attribute} and {node.right.attribute} are constants applied "
                    f"to the operator {node.attribute}"
                )
            return False  # operation of two constants is a constant (unnecessary complexity)
        return check_node_validity(node.left, verbose=verbose) and check_node_validity(
            node.right, verbose=verbose
        )
    else:
        return True


def node_from_prefix(
    prefix_notation: List[str],
    function_test: Callable = lambda _: False,
    operator_test: Callable = lambda _: False,
    variable_test: Callable = lambda _: False,
    constant_test: Callable = lambda _: False,
):
    """
    Create a tree from a prefix notation

    Args:
        prefix_notation: The equation in prefix notation
        function_test: A function that tests if the attribute is a function
        operator_test: A function that tests if the attribute is an operator
        variable_test: A function that tests if the attribute is a variable
        constant_test: A function that tests if the attribute is a constant

    Examples:
        # The equation is a single variable. The test for variables is a check in a hard
        # coded list
        >>> tree = node_from_prefix(['x'], variable_test=lambda x: 'x' in ['x', 'y', 'z'])
        >>> tree.kind == NodeKind.VARIABLE
        True
        >>> tree.attribute
        'x'

        # We can also use more elaborate tests depending. For example checking, weather the
        # attribute contains an underscore
        >>> tree = node_from_prefix(
        ...     prefix_notation=['sin', 'x_1'],
        ...     function_test=lambda x: x in ['sin', 'cos'],
        ...     variable_test=lambda x: '_' in x)
        >>> tree.kind == NodeKind.FUNCTION
        True
        >>> tree.attribute
        'sin'
        >>> tree.left.kind == NodeKind.VARIABLE
        True
        >>> tree.left.attribute
        'x_1'

        # We can also pass other functions
        >>> def is_variable(x):
        ...     return x == 'x' or x == 'y'
        >>> def is_operator(x):
        ...     return not is_variable(x) and len(x) == 1
        >>> def is_function(x):
        ...     return not is_variable(x) and not is_operator(x)
        >>> tree = node_from_prefix(
        ...     prefix_notation=['+', 'sin', 'x', 'cos', 'y'],
        ...     variable_test=is_variable,
        ...     operator_test=is_operator,
        ...     function_test=is_function
        ...     )
        >>> tree.kind == NodeKind.OPERATOR
        True
        >>> tree.attribute
        '+'
        >>> tree.left.kind == NodeKind.FUNCTION
        True
        >>> tree.left.attribute
        'sin'
        >>> tree.left.left.kind == NodeKind.VARIABLE
        True
        >>> tree.left.left.attribute
        'x'
        >>> tree.right.kind == NodeKind.FUNCTION
        True
        >>> tree.right.attribute
        'cos'
        >>> tree.right.left.kind == NodeKind.VARIABLE
        True
        >>> tree.right.left.attribute
        'y'

    """
    node, _ = _from_prefix_recursion(
        prefix_notation, function_test, operator_test, variable_test, constant_test
    )

    return node


def _from_prefix_recursion(
    prefix_notation, function_test, operator_test, variable_test, constant_test, index=0
):
    if index is None or index >= len(prefix_notation):
        return None, None

    attribute = prefix_notation[index]

    if function_test(attribute):
        kind = NodeKind.FUNCTION
    elif operator_test(attribute):
        kind = NodeKind.OPERATOR
    elif variable_test(attribute):
        kind = NodeKind.VARIABLE
    elif (
        constant_test(attribute)
        or is_numeric(attribute)
        or is_known_constant(attribute)
    ):
        kind = NodeKind.CONSTANT
    else:
        raise Exception(f"{attribute} has no defined type in any space")

    if kind == NodeKind.FUNCTION:
        children = 1
    elif kind == NodeKind.OPERATOR:
        children = 2
    else:
        children = 0

    if index - 2 >= len(prefix_notation):
        children = 0

    left_node = None
    right_node = None
    if children >= 1:
        left_node, index = _from_prefix_recursion(
            prefix_notation,
            function_test,
            operator_test,
            variable_test,
            constant_test,
            index + 1,
        )

    if children == 2:
        right_node, index = _from_prefix_recursion(
            prefix_notation,
            function_test,
            operator_test,
            variable_test,
            constant_test,
            index + 1,
        )

    node = TreeNode(left_node, right_node, kind=kind, attribute=attribute)

    return node, index


def sample_attribute(priors: Dict, parent_attribute=""):
    attribute_list = list(priors.keys())
    priors = set_priors(priors, [str(structure) for structure in attribute_list])
    probabilities = [priors[key] for key in priors.keys()]
    sample_index = np.random.choice(len(attribute_list), p=probabilities)
    return attribute_list[sample_index]


def sample_attribute_conditional(priors: Dict, conditional_prior: Dict):
    _priors = priors
    if conditional_prior is not None:
        _priors = conditional_prior
    attribute_list = list(_priors.keys())
    _priors = set_priors(_priors, [str(structure) for structure in attribute_list])
    probabilities = [_priors[key] for key in _priors.keys()]
    sample_index = np.random.choice(len(attribute_list), p=probabilities)
    return attribute_list[sample_index]


def sample_attribute_from_tree(
    tree_structure,
    index,
    feature_priors,
    function_priors,
    operator_priors,
    parent_attribute="",
):
    num_children = _count_children(tree_structure, index)
    if num_children == 0:
        return sample_attribute(feature_priors, parent_attribute)
    elif num_children == 1:
        return sample_attribute(function_priors, parent_attribute)
    elif num_children == 2:
        return sample_attribute(operator_priors, parent_attribute)
    else:
        raise Exception("Invalid number of children: %s" % num_children)


def sample_attribute_from_tree_with_conditionals(
    tree_structure,
    index,
    feature_priors,
    function_priors,
    function_conditionals,
    operator_priors,
    operator_conditionals,
    parent_attribute="",
    parent_kind=NodeKind.NONE,
):
    conditional = None
    cond_prior = None
    if (
        parent_kind == NodeKind.FUNCTION
        and function_conditionals is not None
        and parent_attribute in function_conditionals.keys()
    ):
        conditional = function_conditionals[parent_attribute]
    if (
        parent_kind == NodeKind.OPERATOR
        and operator_conditionals is not None
        and parent_attribute in operator_conditionals.keys()
    ):
        conditional = operator_conditionals[parent_attribute]

    num_children = _count_children(tree_structure, index)

    if num_children == 0:
        if conditional and conditional["features"] != {}:
            cond_prior = conditional["features"]
        return sample_attribute_conditional(feature_priors, cond_prior)
    elif num_children == 1:
        if conditional and conditional["functions"] != {}:
            cond_prior = conditional["functions"]
        return sample_attribute_conditional(function_priors, cond_prior)
    elif num_children == 2:
        if conditional and conditional["operators"] != {}:
            cond_prior = conditional["operators"]
        return sample_attribute_conditional(operator_priors, cond_prior)
    else:
        raise Exception("Invalid number of children: %s" % num_children)


def sample_equation_tree_from_structure(
    tree_structure,
    index,
    feature_priors,
    function_priors,
    operator_priors,
    parent_attribute="",
):
    attribute = sample_attribute_from_tree(
        tree_structure,
        index,
        feature_priors,
        function_priors,
        operator_priors,
        parent_attribute,
    )

    kind = NodeKind.NONE

    if attribute in function_priors.keys():
        kind = NodeKind.FUNCTION
    elif attribute in operator_priors.keys():
        kind = NodeKind.OPERATOR
    elif attribute in feature_priors.keys():
        if "x_" in attribute:
            kind = NodeKind.VARIABLE
        elif "c_" in attribute:
            kind = NodeKind.CONSTANT
        elif "0" in attribute:
            kind = NodeKind.CONSTANT

    node = TreeNode(attribute=attribute, kind=kind)

    children = _get_children(tree_structure, index)

    if len(children) >= 1:
        node.left = sample_equation_tree_from_structure(
            tree_structure,
            children[0],
            feature_priors,
            function_priors,
            operator_priors,
            parent_attribute=attribute,
        )

    if len(children) == 2:
        node.right = sample_equation_tree_from_structure(
            tree_structure,
            children[1],
            feature_priors,
            function_priors,
            operator_priors,
            parent_attribute=attribute,
        )
    return node


def sample_equation_tree_from_structure_with_conditionals(
    tree_structure,
    index,
    feature_priors,
    function_priors,
    function_conditionals,
    operator_priors,
    operator_conditionals,
    parent_attribute="",
    parent_kind=NodeKind.NONE,
):
    attribute = sample_attribute_from_tree_with_conditionals(
        tree_structure,
        index,
        feature_priors,
        function_priors,
        function_conditionals,
        operator_priors,
        operator_conditionals,
        parent_attribute,
        parent_kind,
    )

    num_children = _count_children(tree_structure, index)
    kind = NodeKind.NONE

    if num_children == 1:
        kind = NodeKind.FUNCTION
    elif num_children == 2:
        kind = NodeKind.OPERATOR
    elif attribute is not None:
        if "variable" in attribute:
            kind = NodeKind.VARIABLE
        elif "constant" in attribute:
            kind = NodeKind.CONSTANT

    node = TreeNode(attribute=attribute, kind=kind)

    children = _get_children(tree_structure, index)

    if len(children) >= 1:
        node.left = sample_equation_tree_from_structure_with_conditionals(
            tree_structure,
            children[0],
            feature_priors,
            function_priors,
            function_conditionals,
            operator_priors,
            operator_conditionals,
            parent_attribute=attribute,
            parent_kind=kind,
        )

    if len(children) == 2:
        node.right = sample_equation_tree_from_structure_with_conditionals(
            tree_structure,
            children[1],
            feature_priors,
            function_priors,
            function_conditionals,
            operator_priors,
            operator_conditionals,
            parent_attribute=attribute,
            parent_kind=kind,
        )
    return node


def sample_tree(
    feature_priors={},
    function_priors={},
    operator_priors={},
    structure_priors={},
):
    tree_structure = sample_tree_structure(prior=structure_priors)
    tree = sample_equation_tree_from_structure(
        tree_structure, 0, feature_priors, function_priors, operator_priors
    )
    return tree


def sample_tree_full(prior, max_var_unique):
    """
    Examples:
        >>> np.random.seed(42)

        # We can set priors for features, functions, operators
        # and also conditionals based the parent
        >>> p = {
        ...     'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
        ...     'features': {'constants': .2, 'variables': .8},
        ...     'functions': {'sin': .5, 'cos': .5},
        ...     'operators': {'+': 1., '-': .0},
        ...     'function_conditionals': {
        ...                             'sin': {
        ...                                 'features': {'constants': 0., 'variables': 1.},
        ...                                 'functions': {'sin': 0., 'cos': 1.},
        ...                                 'operators': {'+': 0., '-': 1.}
        ...                             },
        ...                             'cos': {
        ...                                 'features': {'constants': 0., 'variables': 1.},
        ...                                 'functions': {'cos': 1., 'sin': 0.},
        ...                                 'operators': {'+': 0., '-': 1.}
        ...                             }
        ...                         },
        ...     'operator_conditionals': {
        ...                             '+': {
        ...                                 'features': {'constants': 0., 'variables': 1.},
        ...                                 'functions': {'sin': 1., 'cos': 0.},
        ...                                 'operators': {'+': 1., '-': 0.}
        ...                             },
        ...                             '-': {
        ...                                 'features': {'constants': .3, 'variables': .7},
        ...                                 'functions': {'cos': .5, 'sin': .5},
        ...                                 'operators': {'+': .9, '-': .1}
        ...                             }
        ...                         },
        ... }
        >>> sample = sample_tree_full(p, 3)
        >>> sample.attribute
        'cos'
        >>> sample.kind == NodeKind.FUNCTION
        True
        >>> sample.left.attribute
        'cos'
        >>> sample.left.kind == NodeKind.FUNCTION
        True
        >>> sample.right is None
        True
        >>> sample = sample_tree_full(p, 3)
        >>> sample.attribute
        'sin'
        >>> sample.kind == NodeKind.FUNCTION
        True
        >>> sample.left.attribute
        'cos'
        >>> sample.kind == NodeKind.FUNCTION
        True
        >>> sample.left.left.attribute
        'x_1'
        >>> sample.left.left.kind == NodeKind.VARIABLE
        True

        # If we don't provide priors for the conditionals,
        # the fallback is the unconditioned priors
        >>> p = {'max_depth': 8,
        ...     'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
        ...     'features': {'constants': .2, 'variables': .8},
        ...     'functions': {'sin': .5, 'cos': .5},
        ...     'operators': {'+': .5, '-': .5},
        ... }
        >>> sample = sample_tree_full(p, 3)
        >>> sample.attribute
        '-'
        >>> sample.kind == NodeKind.OPERATOR
        True
        >>> sample.right.attribute
        'x_1'
        >>> sample.left.attribute
        'x_1'
        >>> sample.left.kind == NodeKind.VARIABLE
        True

    """
    tree_structure = sample_tree_structure(prior["structures"])
    function_conditionals = None
    operator_conditionals = None
    if "function_conditionals" in prior.keys():
        function_conditionals = prior["function_conditionals"]
    if "operator_conditionals" in prior.keys():
        operator_conditionals = prior["operator_conditionals"]

    tree = sample_equation_tree_from_structure_with_conditionals(
        tree_structure,
        0,
        prior["features"],
        prior["functions"],
        function_conditionals,
        prior["operators"],
        operator_conditionals,
    )
    post(tree, max_var_unique)
    return tree


def sample_tree_full_fast(prior, tree_depth, max_var_unique):
    """
    Examples:
        >>> np.random.seed(42)

        # We can set priors for features, functions, operators
        # and also conditionals based the parent
        >>> p = {
        ...     'features': {'constants': .2, 'variables': .8},
        ...     'functions': {'sin': .5, 'cos': .5},
        ...     'operators': {'+': 1., '-': .0},
        ...     'function_conditionals': {
        ...                             'sin': {
        ...                                 'features': {'constants': 0., 'variables': 1.},
        ...                                 'functions': {'sin': 0., 'cos': 1.},
        ...                                 'operators': {'+': 0., '-': 1.}
        ...                             },
        ...                             'cos': {
        ...                                 'features': {'constants': 0., 'variables': 1.},
        ...                                 'functions': {'cos': 1., 'sin': 0.},
        ...                                 'operators': {'+': 0., '-': 1.}
        ...                             }
        ...                         },
        ...     'operator_conditionals': {
        ...                             '+': {
        ...                                 'features': {'constants': 0., 'variables': 1.},
        ...                                 'functions': {'sin': 1., 'cos': 0.},
        ...                                 'operators': {'+': 1., '-': 0.}
        ...                             },
        ...                             '-': {
        ...                                 'features': {'constants': .3, 'variables': .7},
        ...                                 'functions': {'cos': .5, 'sin': .5},
        ...                                 'operators': {'+': .9, '-': .1}
        ...                             }
        ...                         },
        ... }
        >>> sample = sample_tree_full_fast(p, 3, 2)
        >>> sample.attribute
        'cos'
        >>> sample.kind == NodeKind.FUNCTION
        True
        >>> sample.left.attribute
        'cos'
        >>> sample.left.kind == NodeKind.FUNCTION
        True
        >>> sample.right is None
        True
        >>> sample = sample_tree_full_fast(p, 4, 3)
        >>> sample.attribute
        'sin'
        >>> sample.kind == NodeKind.FUNCTION
        True
        >>> sample.left.attribute
        '-'
        >>> sample.kind == NodeKind.FUNCTION
        True
        >>> sample.left.left.attribute
        'x_1'
        >>> sample.left.left.kind == NodeKind.VARIABLE
        True

        # If we don't provide priors for the conditionals,
        # the fallback is the unconditioned priors
        >>> p = {'max_depth': 8,
        ...     'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
        ...     'features': {'constants': .2, 'variables': .8},
        ...     'functions': {'sin': .5, 'cos': .5},
        ...     'operators': {'+': .5, '-': .5},
        ... }
        >>> sample = sample_tree_full_fast(p, 5, 3)
        >>> sample.attribute
        '-'
        >>> sample.kind == NodeKind.OPERATOR
        True
        >>> sample.right.attribute
        'cos'
        >>> sample.left.attribute
        'sin'
        >>> sample.left.kind == NodeKind.FUNCTION
        True

    """
    tree_structure = sample_tree_structure_fast(tree_depth)
    function_conditionals = None
    operator_conditionals = None
    if "function_conditionals" in prior.keys():
        function_conditionals = prior["function_conditionals"]
    if "operator_conditionals" in prior.keys():
        operator_conditionals = prior["operator_conditionals"]

    tree = sample_equation_tree_from_structure_with_conditionals(
        tree_structure,
        0,
        prior["features"],
        prior["functions"],
        function_conditionals,
        prior["operators"],
        operator_conditionals,
    )
    post(tree, max_var_unique)
    return tree


def post(tree, max_var_unique):
    var_names = [f"unique_{i}" for i in range(1, max_var_unique + 1)]
    c = 1
    v = 1
    var_dict = {}

    def name(node):
        nonlocal c, v, var_dict
        if node is None:
            return
        if node.kind == NodeKind.CONSTANT:
            node.attribute = f"c_{c}"
            c += 1
        if node.kind == NodeKind.VARIABLE:
            _var_name = np.random.choice(var_names)
            if _var_name not in var_dict.keys():
                var_dict[_var_name] = f"x_{v}"
                v += 1
            node.attribute = var_dict[_var_name]
        name(node.left)
        name(node.right)

    name(tree)
