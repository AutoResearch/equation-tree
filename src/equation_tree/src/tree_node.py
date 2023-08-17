from enum import Enum
from typing import Callable, Dict, List

import numpy as np

from equation_tree.src.sample_tree_structure import (
    _count_children,
    _get_children,
    sample_tree_structure,
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
