from typing import Dict, List, Optional, Union

from .tree import EquationTree
from .util.priors import priors_from_space
from .util.type_check import is_constant_formatted, is_variable_formatted

PriorType = Union[List, Dict]

DEFAULT_FUNCTION_SPACE = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
DEFAULT_OPERATOR_SPACE = ["+", "-", "*", "/", "^"]

MAX_ITER = 10_000


def sample_tree_raw(
    max_depth: int = 3,
    max_num_constants: int = 0,
    max_num_variables: int = 1,
    feature_priors: Optional[Dict] = None,
    function_priors: PriorType = DEFAULT_FUNCTION_SPACE,
    operator_priors: PriorType = DEFAULT_OPERATOR_SPACE,
    structure_priors: PriorType = {},
):
    """
    Sample a tree from priors, simplify and check if valid tree
    """
    # Assertions
    if max_depth < 3:
        raise Exception("Can not sample tree with max depth bellow 3")
    if max_num_constants + max_num_variables < 1:
        raise Exception("Can not sample tree without leafs")

    # Get feature priors
    if feature_priors is not None:
        for key in feature_priors:
            if not (is_variable_formatted(key) or is_constant_formatted(key)):
                raise Exception(
                    "Use standard formats for feature priors: x_{int} for variables, "
                    "c_{int} for constants."
                )
        _feature_priors = feature_priors.copy()
    else:
        _feature_space = [f"x_{i}" for i in range(1, max_num_variables + 1)] + [
            f"c_{i}" for i in range(1, max_num_constants + 1)
        ]
        _feature_priors = priors_from_space(_feature_space)

    # Convert priors if space is given
    if isinstance(function_priors, List):
        _function_priors = priors_from_space(function_priors)
    else:
        _function_priors = function_priors.copy()

    if isinstance(operator_priors, List):
        _operator_priors = priors_from_space(operator_priors)
    else:
        _operator_priors = operator_priors.copy()

    if isinstance(structure_priors, List):
        _structure_priors = priors_from_space(structure_priors)
    else:
        _structure_priors = structure_priors.copy()

    # Create tree
    equation_tree = EquationTree.from_priors(
        max_depth=max_depth,
        feature_priors=_feature_priors,
        function_priors=_function_priors,
        operator_priors=_operator_priors,
        structure_priors=_structure_priors,
    )

    # Check if tree is valid
    if not equation_tree.check_validity():
        return None

    equation_tree.simplify(
        function_test=lambda x: x in _function_priors.keys(),
        operator_test=lambda x: x in _operator_priors.keys(),
    )

    # Check is nan
    if equation_tree.is_nan:
        return None

    # Check if duplicate constants
    if (
        equation_tree.n_non_numeric_constants
        > equation_tree.n_non_numeric_constants_unique
    ):
        return None

    # Check if more constants than max:
    if equation_tree.n_constants > max_num_constants:
        return None

    # Check if more variables than max:
    if equation_tree.n_variables > max_num_variables:
        return None

    if not equation_tree.check_validity():
        return None

    if not equation_tree.check_possible(
        _feature_priors, _function_priors, _operator_priors, _structure_priors
    ):
        return None

    equation_tree.get_evaluation()
    if not equation_tree.has_valid_value:
        return None

    return equation_tree


def sample_tree_iter(
    max_depth: int = 3,
    max_num_constants: int = 0,
    max_num_variables: int = 1,
    feature_priors: Optional[Dict] = None,
    function_priors: PriorType = DEFAULT_FUNCTION_SPACE,
    operator_priors: PriorType = DEFAULT_OPERATOR_SPACE,
    structure_priors: PriorType = {},
):
    for _ in range(MAX_ITER):
        equation_tree = sample_tree_raw(
            max_depth,
            max_num_constants,
            max_num_variables,
            feature_priors,
            function_priors,
            operator_priors,
            structure_priors,
        )
        if equation_tree is not None:
            return equation_tree
