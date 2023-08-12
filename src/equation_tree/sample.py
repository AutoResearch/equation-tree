import warnings
from typing import Dict, List, Optional, Union

from equation_tree import EquationTree
from util.priors import priors_from_space

PriorType = Union[List, Dict]

DEFAULT_FUNCTION_SPACE = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
DEFAULT_OPERATOR_SPACE = ["+", "-", "*", "/", "^"]


def sample_tree(
        max_depth: int = 3,
        max_num_constants: int = 0,
        max_num_variables: int = 1,
        feature_priors: Optional[Dict] = None,
        function_priors: PriorType = DEFAULT_FUNCTION_SPACE,
        operator_priors: PriorType = DEFAULT_OPERATOR_SPACE,
        structure_priors: PriorType = {},
):
    # Assertions
    if max_depth < 3:
        raise Exception("Can not sample tree with max depth bellow 3")
    if max_num_constants + max_num_variables < 1:
        raise Exception("Can not sample tree without leafs")

    if feature_priors is not None:
        for key in feature_priors:
            if key

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

    # Create feature_priors

    equation_tree = EquationTree.from_priors(
        max_depth=max_depth,
        feature_priors=feature_priors,
        function_priors=function_priors,
        operator_priors=operator_priors,
        structure_priors=structure_priors,
    )

#
# def sample_trees(
#     num_samples: int,
#     max_depth: int,
#     max_num_variables: int,
#     max_num_constants: int,
#     function_space: list = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"],
#     operator_space: list = ["+", "-", "*", "/", "^"],
#     function_priors: dict = {},
#     operator_priors: dict = {},
#     force_full_domain: bool = False,
#     with_replacement: bool = True,
#     fix_num_variables_to_max: bool = False,
#     include_zero_as_constant=False,
#     min_input_value: float = -1,
#     max_input_value: float = 1,
#     min_constant_value: float = -1,
#     max_constant_value: float = 1,
#     num_input_points: int = 100,
#     num_constant_points: int = 100,
#     num_evaluation_samples: int = 100,
#     max_iter: int = 1000000,
#     num_burns: int = 0,
#     require_simplify: bool = True,
#     is_real_domain: bool = True,
#     verbose: bool = False,
# ):
#     """
#     Generate data for the equation generator.
#
#     Arguments:
#         num_samples: Number of samples to generate.
#         max_depth: Maximum depth of the equation tree.
#         max_num_variables: Number of variables in the equation tree.
#         max_num_constants: Maximum number of constants in the equation tree.
#         function_space: List of functions to use in the equation tree.
#         operator_space: List of operations to use in the equation tree.
#         function_priors: Dict with priors for the functions.
#         operator_priors: Dict with priors for the operators.
#         force_full_domain: If true only equations that are defined on full R are sampled.
#         with_replacement: Whether to sample with replacement.
#         fix_num_variables_to_max: Whether to fix the number of variables.
#         include_zero_as_constant: Whether to include zero as a constant.
#         min_input_value: Minimum value of the input variables.
#         max_input_value: Maximum value of the input variables.
#         min_constant_value: Minimum value of the constants.
#         max_constant_value: Maximum value of the constants.
#         num_input_points: Number of points to sample for each input variable and constant.
#         num_constant_points: Number of points to sample for each constant.
#         num_evaluation_samples: ...,
#         max_iter: ...,
#         num_burns: Number of times before sampling to adjust the probabilities
#         require_simplify: Defines if the equations are simplified
#         is_real_domain: Defines if the variables and constants are real or complex numbers
#         verbose: Defines if additional output is generated
#     """
#     local_dict = locals()
#     del local_dict["num_burns"]
#     del local_dict["verbose"]
#     del local_dict["num_samples"]
#     del local_dict["num_evaluation_samples"]
#     del local_dict["operator_priors"]
#     del local_dict["function_priors"]
#
#     if max_depth < 3:
#         raise ValueError("max_depth must be at least 3")
#     if force_full_domain:
#         warnings.warn(
#             "Forcing equations to be defined on full domain may lead "
#             "to larger discrepancies between priors and frequencies in sample."
#         )
#
#     tokenized_equation_list: List[List[str]] = list()
#     sympy_equation_list: List[str] = list()
#     evaluation_list: List[float] = list()
#     max_equation_elements = 0
#
#     # Generate Feature Space
#     feature_space = [f"x_{i + 1}" for i in range(max_num_variables)] + [
#         f"c_{i + 1}" for i in range(max_num_constants)
#     ]
#
#     if include_zero_as_constant:
#         feature_space.append("0")
#
#     # Get the configuration
#     config = {
#         "max_num_variables": max_num_variables,
#         "max_num_constants": max_num_constants,
#         "feature_space": feature_space,
#         "function_space": function_space,
#         "operator_space": operator_space,
#         "force_full_domain": force_full_domain,
#         "with_replacement": with_replacement,
#         "fix_num_variables_to_max": fix_num_variables_to_max,
#         "include_zero_as_constant": include_zero_as_constant,
#         "min_input_value": min_input_value,
#         "max_input_value": max_input_value,
#         "min_constant_value": min_constant_value,
#         "max_constant_value": max_constant_value,
#         "num_input_points": num_input_points,
#         "num_constant_points": num_constant_points,
#         "num_evaluation_samples": num_evaluation_samples,
#         "max_iter": max_iter,
#         "require_simplify": require_simplify,
#         "is_real_domain": is_real_domain,
#         "verbose": verbose,
#     }
#
#     # Generate all possible trees
#     tree_structures = [
#         tree.copy()
#         for depth in range(3, max_depth + 1)
#         for tree in rooted_tree_iterator(depth)
#     ]
#
#     # set target priors
#     target_probabilities_functions = _set_priors(function_priors, function_space)
#     target_probabilities_operators = _set_priors(operator_priors, operator_space)
#
#     hash_id_backup = str(local_dict)
#     if function_priors != {} or operator_priors != {}:
#         local_dict["functon_priors"] = target_probabilities_functions
#         local_dict["operator_priors"] = target_probabilities_operators
#         if num_burns > 0:
#             warnings.warn(
#                 f"Storing non default priors. "
#                 f"Please make sure BURN_SAMPLE_SIZE {BURN_SAMPLE_SIZE} is large enough"
#             )
#     hash_id = str(local_dict)
#
#     # load adjusted probabilities from hash
#     function_probabilities, operator_probabilities = load_adjusted_probabilities(
#         hash_id
#     )
#
#     if function_probabilities is None:
#         function_probabilities_, operator_probabilities_ = load_adjusted_probabilities(
#             hash_id_backup
#         )
#         if function_probabilities_ is not None:
#             warnings.warn(
#                 "Load backup probabilities from default priors and adjusting them. "
#                 "This may lead to discrepancies between priors and sampled frequencies."
#             )
#             _f = {
#                 key: target_probabilities_functions[key]
#                 * len(target_probabilities_functions)
#                 for key in target_probabilities_functions.keys()
#             }
#             _o = {
#                 key: target_probabilities_operators[key]
#                 * len(target_probabilities_operators)
#                 for key in target_probabilities_operators.keys()
#             }
#             function_probabilities = {
#                 key: function_probabilities_[key] * _f[key]
#                 for key in target_probabilities_functions.keys()
#             }
#             operator_probabilities = {
#                 key: operator_probabilities_[key] * _o[key]
#                 for key in target_probabilities_operators.keys()
#             }
#         else:
#             function_probabilities = target_probabilities_functions.copy()
#             operator_probabilities = target_probabilities_operators.copy()
#             if num_burns <= 0:
#                 warnings.warn(
#                     "Using raw priors without burn. This may lead to discrepancies "
#                     "between priors and sampled frequencies."
#                 )
#             else:
#                 warnings.warn(
#                     "Using raw priors. This may lead to discrepancies "
#                     "between priors and sampled frequencies."
#                 )
#     # burn (sample and adjust on samples)
#     for burn in range(num_burns):
#         tokenized_equation_list_burn: List[List[str]] = list()
#         # sample an equation
#         for i in range(BURN_SAMPLE_SIZE):
#             tokenized_equation_burn, _, __ = _sample_full_equation(
#                 tree_structures,
#                 tokenized_equation_list_burn,
#                 function_probabilities,
#                 operator_probabilities,
#                 **config,
#             )
#             tokenized_equation_list_burn.append(tokenized_equation_burn)
#
#             nr_burns = burn * BURN_SAMPLE_SIZE + i + 1
#
#             # print progress
#             if nr_burns % PRINT_MOD == 0:
#                 print(f"{nr_burns} equations burned")
#         function_frequencies, operator_frequencies = get_frequencies(
#             tokenized_equation_list_burn, **config
#         )
#
#         for key in function_space:
#             if target_probabilities_functions[key] > 0.0 >= function_frequencies[key]:
#                 warnings.warn(
#                     f"Target probability is greater then 0. but function frequency is 0. "
#                     f"The settings for the sample might not allow "
#                     f'for function "{key}" to be sampled.'
#                 )
#             diff = target_probabilities_functions[key] - function_frequencies[key]
#             if (
#                 function_frequencies[key] > 0.0
#                 or target_probabilities_functions[key] <= 0.0
#             ):
#                 function_probabilities[key] += LEARNING_RATE * diff
#             if function_probabilities[key] <= 0:
#                 function_probabilities[key] = 0
#         for key in operator_space:
#             if target_probabilities_operators[key] > 0.0 >= operator_frequencies[key]:
#                 warnings.warn(
#                     f"Target probability is greater then 0. but function frequency is 0. "
#                     f"The settings for the sample might not allow "
#                     f'for operator "{key}" to be sampled.'
#                 )
#
#             diff = target_probabilities_operators[key] - operator_frequencies[key]
#             if (
#                 operator_frequencies[key] > 0.0
#                 or target_probabilities_operators[key] <= 0.0
#             ):
#                 operator_probabilities[key] += LEARNING_RATE * diff
#             if operator_probabilities[key] <= 0:
#                 operator_probabilities[key] = 0
#         function_probabilities = _normalize_priors(function_probabilities)
#         operator_probabilities = _normalize_priors(operator_probabilities)
#         if SAVE_TO_HASH:
#             print("****")
#             print(function_frequencies, operator_frequencies)
#             print("****")
#             store_adjusted_probabilities(
#                 hash_id, function_probabilities, operator_probabilities
#             )
#
#     # sample
#     for sample in range(num_samples):
#         # sample an equation
#         tokenized_equation, sympy_expression, evaluation = _sample_full_equation(
#             tree_structures,
#             tokenized_equation_list,
#             function_probabilities,
#             operator_probabilities,
#             **config,
#         )
#
#         # add to lists
#         tokenized_equation_list.append(tokenized_equation)
#         sympy_equation_list.append(sympy_expression)
#         evaluation_list.append(evaluation)
#         max_equation_elements = max(max_equation_elements, len(tokenized_equation))
#
#         # print progress
#         if (sample + 1) % PRINT_MOD == 0:
#             print(f"{sample + 1} equations generated")
#
#     # pad the equations and evaluations
#     for idx, equation in enumerate(tokenized_equation_list):
#         num_equation_elements = len(equation)
#         for i in range(max_equation_elements - num_equation_elements):
#             tokenized_equation_list[idx].append(padding)
#             evaluation_list[idx] = np.append(
#                 evaluation_list[idx], np.zeros((num_evaluation_samples, 1)), axis=1
#             )
#
#     # transpose each evaluation
#     # (this is temporary to work with the autoencoder model and may be removed in the future)
#     for idx, evaluation in enumerate(evaluation_list):
#         evaluation_list[idx] = evaluation.T
#
#     print("all equations generated")
#     return {
#         "tokenized_equations": tokenized_equation_list,
#         "sympy_equations": sympy_equation_list,
#         "evaluations": evaluation_list,
#     }
