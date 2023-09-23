# import numpy as np
# import pandas as pd
# from sympy import sympify
# import sympy
# import pickle

from equation_tree import burn
# from equation_tree import sample
from playground.physics_piriors import prior
# from equation_tree.defaults import (
#     functions_prior,
#     is_function,
#     is_operator,
#     operators_prior,
# )
# from equation_tree.sample import sample_fast
# from equation_tree.tree import EquationTree
# from equation_tree.prior import normalize

# p = prior.copy()
# normalize(p)
#
# print(p)


#
#
#
# eq = sample(10000, p, 100)
# print(eq)
# print(prior)
#
import random
for _ in range(5):
    burn(
        prior,
        100,
        "../src/equation_tree/data/_hashed_probabilities.json",
        10_000,
        0.5,
    )


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

# expr = '3*x_1+(sin(x_2))'
# equation_tree = EquationTree.from_sympy(sympy.sympify(expr), is_function, is_operator,
#                                         lambda x: x in ['x_1', 'x_2'], lambda x: x in ['3'])
# equation_tree.draw_tree('test.gv')

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
