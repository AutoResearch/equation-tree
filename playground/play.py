# from sympy import sympify
import random

import numpy as np

from equation_tree.sample import sample
from equation_tree.tree import instantiate_constants

# from equation_tree.tree import EquationTree

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

prior = {
    "structures": {"[0, 1, 1]": 1.0},
    "functions": {"sin": 0.45, "cos": 0.45, "tan": 0.1},
    "operators": {"+": 0.1, "*": 0.1, "^": 0.1, "/": 0.7, "-": 0.0},
    "features": {"variables": 0.1, "constants": 0.9},
    "function_conditionals": {
        "cos": {"features": {"variables": 1.0}, "functions": {"sin": 1.0}},
        "sin": {"features": {"variables": 1.0}, "functions": {"cos": 1.0}},
    },
}

np.random.seed(42)
t = sample(5, prior, 1000)[0]

print(t.sympy_expr)
t_instantiated = instantiate_constants(t, lambda: random.random())
print(t_instantiated.sympy_expr)
