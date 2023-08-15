from equation_tree.tree import EquationTree
from sympy import sympify

expr = sympify('B*x_1**2')
print(expr)
is_operator = lambda x: x in ['+', '*', '**']
is_function = lambda x: x in ['sin']
is_variable = lambda x: '_' in x
is_constant = lambda x: x == 'B'
equation_tree = EquationTree.from_sympy(
    sympify('B*x_1**2'),
    operator_test=is_operator,
    variable_test=is_variable,
    constant_test=is_constant,
    function_test=is_function
)

print(equation_tree.sympy_expr)
