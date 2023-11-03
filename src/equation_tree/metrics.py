from typing import Union

import pandas as pd
import numpy as np

from sympy import sympify, simplify

from equation_tree.tree import EquationTree, ned


def normalized_tree_distance(e_a: EquationTree, e_b: EquationTree):
    """
    Normalized edit distance between two trees according to
    `Li, Y., & Chenguang, Z. (2011). A metric normalization of tree edit distance.
    Frontiers of Computer Science in China, 5, 119-125.`

    """
    if e_a.root is None or e_b.root is None:
        return 1
    return ned(e_a.root, e_b.root)


def prediction_distance(
        e_a: EquationTree, e_b: EquationTree, X: Union[dict, pd.DataFrame]
):
    """
    Mean squared difference between the prediction of two equations on X

    Examples:
        >>> is_operator = lambda x : x in ['+', '*', '**', '-']
        >>> is_variable = lambda x : x in ['x', 'y']
        >>> expr_1 = sympify('x + y')
        >>> expr_2 = sympify('x')
        >>> et_1 = EquationTree.from_sympy(
        ...     expr_1,
        ...     operator_test=is_operator,
        ...     variable_test=is_variable,
        ... )
        >>> et_1.sympy_expr
        x_1 + x_2
        >>> et_2 = EquationTree.from_sympy(
        ...     expr_2,
        ...     operator_test=is_operator,
        ...     variable_test=is_variable,
        ... )
        >>> et_2.sympy_expr
        x_1
        >>> prediction_distance(et_1, et_2, {'x_1': [1], 'x_2': [1]})
        1.0
        >>> prediction_distance(et_1, et_2, {'x_1': [1, 2, 3], 'x_2': [0, 0, 0]})
        0.0
        >>> prediction_distance(et_1, et_2, {'x_1': [1, 2], 'x_2': [1, 2]})
        2.5

    """
    predict_a = e_a.evaluate(X)
    predict_b = e_b.evaluate(X)
    squared_diff = (predict_a - predict_b) ** 2
    return squared_diff.mean()


def symbolic_solution_diff(e_a: EquationTree, e_b: EquationTree):
    """
    Symbolic solution with difference constant based on
    `La Cava, W. et al (2021).
    Contemporary symbolic regression methods and their relative performance.`
    Examples:
        >>> is_operator = lambda x : x in ['+', '*', '**', '-']
        >>> is_variable = lambda x : x in ['x', 'y']
        >>> is_constant = lambda x: is_numeric(x)
        >>> expr_1 = sympify('x + .1')
        >>> expr_2 = sympify('x')
        >>> et_1 = EquationTree.from_sympy(
        ...     expr_1,
        ...     operator_test=is_operator,
        ...     variable_test=is_variable,
        ...     constant_test=is_constant
        ... )
        >>> et_1.sympy_expr
        x_1 + 0.1
        >>> et_2 = EquationTree.from_sympy(
        ...     expr_2,
        ...     operator_test=is_operator,
        ...     variable_test=is_variable,
        ... )
        >>> et_2.sympy_expr
        x_1
        >>> symbolic_solution_diff(et_1, et_2)
        0.1
    """
    diff = simplify(e_a.sympy_expr - e_b.sympy_expr)
    if diff.is_constant():
        return float(diff)
    else:
        return np.infty


def symbolic_solution_quot(e_a: EquationTree, e_b: EquationTree):
    """
    Symbolic solution with quotient constant based on
    `La Cava, W. et al (2021).
    Contemporary symbolic regression methods and their relative performance.`
    Examples:
        >>> is_operator = lambda x : x in ['+', '*', '**', '-']
        >>> is_variable = lambda x : x in ['x', 'y']
        >>> is_constant = lambda x: is_numeric(x)
        >>> expr_1 = sympify('x * .1')
        >>> expr_2 = sympify('x')
        >>> et_1 = EquationTree.from_sympy(
        ...     expr_1,
        ...     operator_test=is_operator,
        ...     variable_test=is_variable,
        ...     constant_test=is_constant
        ... )
        >>> et_1.sympy_expr
        0.1*x_1
        >>> et_2 = EquationTree.from_sympy(
        ...     expr_2,
        ...     operator_test=is_operator,
        ...     variable_test=is_variable,
        ... )
        >>> et_2.sympy_expr
        x_1
        >>> symbolic_solution_quot(et_1, et_2)
        0.1
    """
    quot_1 = simplify(e_a.sympy_expr / e_b.sympy_expr)
    quot_2 = simplify(e_a.sympy_expr / e_b.sympy_expr)
    if quot_1.is_constant():
        return min(float(quot_1), float(quot_2))
    else:
        return np.infty
