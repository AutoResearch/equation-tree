import re

from sympy import Add, Function, Mul, Pow

DEFAULT_CONSTANTS = ["e", "pi"]


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
    pattern = r"^x_\d+$"
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
    pattern = r"^c_\d+$"
    return re.match(pattern, s) is not None


def is_known_constant(s: str):
    return s in DEFAULT_CONSTANTS


def check_functions(expression, function_test):
    """
    Example:
        >>> from sympy import sympify
        >>> expr = sympify('sin(x)')
        >>> check_functions(expr, lambda x: x in ['sin'])
        True

        >>> expr = sympify('x + y')
        >>> check_functions(expr, lambda x: x in ['sin'])
        True

        >>> expr = sympify('sin(x) + cos(y)')
        >>> check_functions(expr, lambda x: x in ['sin', 'cos'])
        True

        >>> check_functions(expr, lambda x: x in ['sin'])
        False

    """

    def apply_test(node):
        if isinstance(node, Function):
            return function_test(str(node.func).lower()) or str(node.func) == "re"
        elif isinstance(node, (Add, Mul, Pow)):
            return all(apply_test(arg) or str(node.func) == "re" for arg in node.args)
        return True

    return apply_test(expression)


def parse_string_list_int(lst):
    """
    Example:
        >>> a = '[1, 2, 3, 4]'
        >>> parse_string_list_int(a)
        [1, 2, 3, 4]
        >>> b = '[10, 2, 3]'
        >>> parse_string_list_int(b)
        [10, 2, 3]
    """
    res = []
    i = 0
    while i < len(lst) - 1:
        start = i
        end = i
        while is_numeric(lst[i]):
            end += 1
            i += 1
        if start != end:
            i = end
            res.append(int(lst[start:end]))
        i += 1
    return res
