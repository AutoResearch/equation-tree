from enum import Enum
from typing import Dict, Callable, List

import numpy as np
from src.sample_tree_structure import _count_children, _get_children, sample_tree_structure
from util.priors import set_priors
from util.type_check import is_numeric

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

    def check_validity(self,
                       zero_representations=['0'],
                       log_representations=['log', 'Log'],
                       division_representations=['/', ':'],
                       verbose=False):
        return check_node_validity(self,
                                   zero_representations,
                                   log_representations,
                                   division_representations,
                                   verbose)


def check_node_validity(node=None,
                        zero_representations=['0'],
                        log_representations=['log', 'Log'],
                        division_representations=['/', ':'],
                        verbose=False):
    if node is None:
        return True

    if node.kind == NodeKind.FUNCTION:
        if node.left is None or node.right is not None:
            return False
        if node.attribute in log_representations and node.left in zero_representations:
            if verbose:
                print(
                    "logarithm is applied to 0 which is results in not real number."
                )
            return False
        if node.left.kind == NodeKind.CONSTANT:  # unnecessary complexity
            if verbose:
                print(
                    f"{node.left.attribute} is a constant "
                    f"applied to a function {node.attribute}"
                )
            return False
        return check_node_validity(node.left, verbose)

    elif node.kind == NodeKind.OPERATOR:
        if node.left is None or node.right is None:
            return False
        if (node.attribute in division_representations and
                node.right.attribute in zero_representations):
            if verbose:
                print("division by 0 is not allowed.")
            return False
        if (
                node.left.kind == NodeKind.CONSTANT
                and node.right.kind == NodeKind.CONSTANT
        ):
            if verbose:
                print(
                    f"{node.left.attribute} and {node.right.attribute} are constants applied "
                    f"to the operator {node.attribute}"
                )
            return False  # operation of two constants is a constant (unnecessary complexity)
        return check_node_validity(node.left, verbose) and check_node_validity(node.right, verbose)
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
    node, _ = _from_prefix_recursion(prefix_notation,
                                     function_test,
                                     operator_test,
                                     variable_test,
                                     constant_test)

    return node


def _from_prefix_recursion(prefix_notation,
                           function_test,
                           operator_test,
                           variable_test,
                           constant_test,
                           index=0):
    attribute = prefix_notation[index]

    if function_test(attribute):
        kind = NodeKind.FUNCTION
    elif operator_test(attribute):
        kind = NodeKind.OPERATOR
    elif variable_test(attribute):
        kind = NodeKind.VARIABLE
    elif constant_test(attribute) or is_numeric(attribute):
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
            index + 1
        )

    if children == 2:
        right_node, index = _from_prefix_recursion(
            prefix_notation,
            function_test,
            operator_test,
            variable_test,
            constant_test,
            index + 1
        )

    node = TreeNode(left_node, right_node, kind=kind, attribute=attribute)

    return node, index


def sample_attribute(priors: Dict, parent_attribute=""):
    attribute_list = list(priors.keys())
    priors = set_priors(priors, [str(structure) for structure in attribute_list])
    probabilities = [priors[key] for key in priors.keys()]
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


def sample_tree(
        max_depth,
        feature_priors={},
        function_priors={},
        operator_priors={},
        structure_priors={},
):
    tree_structure = sample_tree_structure(max_depth=max_depth, priors=structure_priors)
    tree = sample_equation_tree_from_structure(
        tree_structure, 0, feature_priors, function_priors, operator_priors
    )
    return tree
