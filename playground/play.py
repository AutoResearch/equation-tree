# from sympy import sympify

from equation_tree.sample import burn  # , sample

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
    "structures": {},
    "functions": {"sin": 0.45, "cos": 0.45, "tan": 0.1},
    "operators": {"+": 0.1, "*": 0.1, "^": 0.1, "/": 0.7, "-": 0.0},
    "features": {"variables": 0.1, "constants": 0.9},
    "function_conditionals": {
        "cos": {"features": {"variables": 1.0}, "functions": {"sin": 1.0}},
        "sin": {"features": {"variables": 1.0}, "functions": {"cos": 1.0}},
    },
}

burn(prior, 3, "temp.json", 1000)
