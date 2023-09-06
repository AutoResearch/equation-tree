from prior import (
    DEFAULT_FUNCTION_SPACE,
    DEFAULT_OPERATOR_SPACE,
    DEFAULT_PRIOR_FUNCTIONS,
    DEFAULT_PRIOR_OPERATORS,
    structure_prior_from_depth,
    structure_prior_from_max_depth,
)

operator_space = DEFAULT_OPERATOR_SPACE
function_space = DEFAULT_FUNCTION_SPACE
operators_prior = DEFAULT_PRIOR_OPERATORS
functions_prior = DEFAULT_PRIOR_FUNCTIONS
structure_prior_from_depth = structure_prior_from_depth
structure_prior_from_leaves = structure_prior_from_max_depth


def is_operator(a):
    return a in operator_space


def is_function(a):
    return a in function_space
