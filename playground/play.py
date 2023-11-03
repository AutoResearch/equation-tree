# import numpy as np
# import pandas as pd
from equation_tree import sample, EquationTree
from equation_tree.sample import burn
from equation_tree.tree import instantiate_constants
from equation_tree.prior import structure_prior_from_max_depth, \
    priors_from_space
import numpy as np
from sympy import sympify

expr = sympify('x_1**2 + x_2')

is_operator = lambda x: x in ['*', '/', '**', '+']
is_variable = lambda x: x in ['x_1', 'x_2']
is_function = lambda x: x in ['sin']
equation_tree = EquationTree.from_sympy(expr, function_test=is_function)
print(equation_tree.expr)


