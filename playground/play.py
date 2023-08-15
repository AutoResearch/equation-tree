from sympy import sympify

from equation_tree.tree import EquationTree

from equation_tree.sample import sample_tree_raw

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
    'max_depth': 10,
    'structure': {'[0, 1, 2, 1, 2]': .5, '[0, 1, 2, 1]': .5},
    'functions': {'sin': .5, 'cos': .5},
    'operators': {'+': .1, '*': .1, '^': .1, '/': .7},
    'features': {'variables': .1, 'constants': .9},
    'function_conditionals':
        {'cos':
             {'features': {'variables': 1.},
              'functions': {'sin': 1.}},
         'sin':
             {'features': {'variables': 1.},
              'functions': {'cos': 1.}},
         }

}

t = sample_tree_raw(prior, 2)
while t is None:
    t = sample_tree_raw(prior, 2)

