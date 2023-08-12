import re


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
