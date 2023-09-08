# import numpy as np
# import pandas as pd
# from sympy import sympify

from equation_tree.defaults import functions_prior, operators_prior
from equation_tree.sample import sample_fast

# from equation_tree.tree import EquationTree

# import random


# from equation_tree.tree import instantiate_constants


# expr = sympify("B*x_1**2")
# print(expr)
# is_operator = lambda x: x in ["+", "*", "**"]
# is_function = lambda x: x in ["sin"]
# is_variable = lambda x: "_" in x
# is_constant = lambda x: x == "B"
# equation_tree = EquationTree.from_sympy(
#     sympify("B*x_1**2"),
#     operator_test=is_operator,
#     variable_test=is_variable,
#     constant_test=is_constant,
#     function_test=is_function,
# )
#
# print(equation_tree.sympy_expr)

res = sample_fast(
    1,
    {
        "functions": functions_prior,
        "operators": operators_prior,
        "features": {"constants": 0.5, "variables": 0.5},
    },
    50,
    5,
)
print(res[0].sympy_expr)

# expr = sympify("x_a + 3 * y")
#
#
# def is_operator(x):
#     return x in ["+", "*"]
#
#
# def is_variable(x):
#     return "_" in x or x in ["y"]
#
#
# equation_tree = EquationTree.from_sympy(
#     expr, operator_test=is_operator, variable_test=is_variable
# )
# equation_tree.export_to_srbench("test")
