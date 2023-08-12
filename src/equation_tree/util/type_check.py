import re
from sympy import symbols, Function, Add, Mul, Pow

DEFAULT_CONSTANTS = ['e', 'pi']


def is_numeric(s):
    """
    Tests weather the input is a number

    Examples:
        >>> is_numeric('0')
        True
        >>> is_numeric('10')
        True
        >>> is_numeric('hallo')
        False
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_variable_formatted(s: str):
    """
    Tests weather the input is in standard format for a variable

    Examples:
        >>> is_variable_formatted('x_1')
        True
        >>> is_variable_formatted('x_42')
        True
        >>> is_variable_formatted('x_a')
        False
        >>> is_variable_formatted('abc')
        False
        >>> is_variable_formatted('c_3')
        False
    """
    pattern = r'^x_\d+$'
    return re.match(pattern, s) is not None


def is_constant_formatted(s: str):
    """
        Tests weather the input is in standard format for a constant

        Examples:
            >>> is_constant_formatted('c_4')
            True
            >>> is_constant_formatted('c_42')
            True
            >>> is_constant_formatted('c_a')
            False
            >>> is_constant_formatted('abc')
            False
            >>> is_constant_formatted('x_3')
            False
        """
    pattern = r'^c_\d+$'
    return re.match(pattern, s) is not None


def is_known_constant(s: str):
    return s in DEFAULT_CONSTANTS


def check_functions(expression, function_test):
    """
    Example:
        >>> from sympy import sympify
        >>> expr = sympify('sin(x)')
        >>> contains_function(expr, lambda x: x in ['sin'])
        True

        >>> expr = sympify('x + y')
        >>> contains_function(expr, lambda x: x in ['sin'])
        True

        >>> expr = sympify('sin(x) + cos(y)')
        >>> contains_function(expr, lambda x: x in ['sin', 'cos'])
        True

        >>> contains_function(expr, lambda x: x in ['sin'])
        False

    """

    def apply_test(node):
        if isinstance(node, Function):
            return function_test(str(node.func).lower()) or str(node.func) == 're'
        elif isinstance(node, (Add, Mul)):
            return all(apply_test(arg) or str(node.func) == 're' for arg in node.args)
        return True

    return apply_test(expression)

    #return not has_any_function(expression) or has_function(expression, function_test)
