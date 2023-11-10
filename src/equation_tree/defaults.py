from equation_tree.prior import (
    # DEFAULT_FUNCTION_SPACE,
    # DEFAULT_OPERATOR_SPACE,
    # DEFAULT_PRIOR_FUNCTIONS,
    # DEFAULT_PRIOR_OPERATORS,
    priors_from_space,
    structure_prior_from_depth,
    structure_prior_from_max_depth,
)

from equation_tree.util.type_check import is_numeric, is_known_constant, \
    is_constant_formatted, is_variable_formatted

DEFAULT_FUNCTION_SPACE = [
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "abs",
    "acos",
    "asin"
]

ADDITIONAL_FUNCTIONS = [
    "arg",
    "sinh",
    "cosh",
    "tanh",
    "cot"
]

DEFAULT_OPERATOR_SPACE = ["+", "-", "*", "/", "**", "max", "min"]

DEFAULT_PRIOR_FUNCTIONS = priors_from_space(DEFAULT_FUNCTION_SPACE)
DEFAULT_PRIOR_OPERATORS = priors_from_space(DEFAULT_OPERATOR_SPACE)

operator_space = DEFAULT_OPERATOR_SPACE
function_space = DEFAULT_FUNCTION_SPACE
operators_prior = DEFAULT_PRIOR_OPERATORS
functions_prior = DEFAULT_PRIOR_FUNCTIONS
features_prior = {'constants': 0.5, 'variables': 0.5}
structure_prior_from_depth = structure_prior_from_depth
structure_prior_from_leaves = structure_prior_from_max_depth

DEFAULT_PRIOR = {
    'structures': structure_prior_from_max_depth(6),
    'functions': functions_prior,
    'operators': operators_prior,
    'features': features_prior
}


def is_operator(a):
    return a.lower() in operator_space


def is_function(a):
    return a.lower() in function_space


def is_variable(a):
    return is_variable_formatted(a)


def is_constant(a):
    return is_constant_formatted(a) or is_numeric(a) or is_known_constant(a)
