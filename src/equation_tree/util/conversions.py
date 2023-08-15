import re

from sympy import symbols

from .type_check import is_numeric


def prefix_to_infix(
    prefix, function_test=lambda _: False, operator_test=lambda _: False
):
    """
    Transforms prefix notation to infix notation

    Example:
        >>> is_function = lambda x: x in ['sin', 'cos']
        >>> is_operator = lambda x : x in ['+', '-', '*']
        >>> prefix_to_infix(['-', 'x_1', 'x_2'], is_function, is_operator)
        '(x_1-x_2)'

        >>> prefix_to_infix(
        ...     ['*', 'x', 'cos', '+', 'y', 'z'], is_function, is_operator)
        '(x*cos((y+z)))'

    """
    stack = []
    for i in range(len(prefix) - 1, -1, -1):
        if function_test(prefix[i]):
            # symbol in unary operator
            stack.append(prefix[i] + "(" + stack.pop() + ")")
        elif operator_test(prefix[i]):
            # symbol is binary operator
            str = "(" + stack.pop() + prefix[i] + stack.pop() + ")"
            stack.append(str)
        else:
            # symbol is operand
            stack.append(prefix[i])
    return stack.pop()


def infix_to_prefix(infix, function_test, operator_test):
    """
    Transforms prefix notation to infix notation

    Example:
        >>> is_function = lambda x: x in ['sin', 'cos']
        >>> is_operator = lambda x : x in ['+', '-', '*']
        >>> infix_to_prefix('x_1-x_2', is_function, is_operator)
        ['-', 'x_1', 'x_2']


        >>> infix_to_prefix(
        ...     'x_1*cos(c_1+x_2)', is_function, is_operator)
        ['*', 'x_1', 'cos', '+', 'c_1', 'x_2']

    """
    n = len(infix)

    infix = list(infix[::-1].lower())

    for i in range(n):
        if infix[i] == "(":
            infix[i] = ")"
        elif infix[i] == ")":
            infix[i] = "("

    infix = "".join(infix)
    postfix = _infix_to_postfix(infix, function_test, operator_test)
    prefix = postfix[::-1]

    return prefix


def standardize_sympy(
    sympy_expr, variable_test=lambda _: False, constant_test=lambda _: False
):
    """
    replace all variables and constants with standards

    Example:
        >>> from sympy import sympify
        >>> expr = sympify('x + A * cos(z+y)')
        >>> expr
        A*cos(y + z) + x

        >>> is_variable = lambda x : x in ['x', 'y', 'z']
        >>> is_constant = lambda x : x in ['A']
        >>> standardize_sympy(expr, is_variable, is_constant)
        c_1*cos(x_2 + x_3) + x_1

        >>> expr = sympify('x_a+B*y')
        >>> expr
        B*y + x_a
        >>> is_variable = lambda x : '_' in x or x in ['y']
        >>> is_constant = lambda x : x == 'B'
        >>> standardize_sympy(expr, is_variable, is_constant)
        c_1*x_2 + x_1

        >>> expr = sympify('x ** x')
        >>> expr
        x**x
        >>> is_variable = lambda x: x in ['x']
        >>> standardize_sympy(expr, is_variable)
        x_1**x_1

        >>> expr = sympify('sin(C*x) + cos(C*x)')
        >>> expr
        sin(C*x) + cos(C*x)

        >>> is_variable = lambda x: x == 'x'
        >>> is_constant = lambda x: x == 'C'
        >>> standardize_sympy(expr, is_variable, is_constant)
        sin(c_1*x_1) + cos(c_1*x_1)
    """
    variable_count = 0
    constant_count = 0
    variables = {}
    constants = {}

    def replace_symbols(node):
        nonlocal variable_count, constant_count, variables, constants
        if variable_test(str(node)):
            if not str(node) in variables.keys():
                variable_count += 1
                new_symbol = symbols(f"x_{variable_count}")
                variables[str(node)] = new_symbol
            else:
                new_symbol = variables[str(node)]
            return new_symbol
        elif constant_test(str(node)) and not is_numeric(str(node)):
            if not str(node) in constants.keys():
                constant_count += 1
                new_symbol = symbols(f"c_{constant_count}")
                constants[str(node)] = new_symbol
            else:
                new_symbol = constants[str(node)]
            return new_symbol
        else:
            return node

    def recursive_replace(node):
        if node.is_Function or node.is_Add or node.is_Mul or node.is_Pow:
            return node.func(*[recursive_replace(arg) for arg in node.args])
        return replace_symbols(node)

    new_expression = recursive_replace(sympy_expr)
    return new_expression


def unary_minus_to_binary(expr, operator_test):
    """
    replace unary minus with binary

    Examples:
        >>> o = lambda x: x in ['+', '-', '*', '/', '^']
        >>> unary_minus_to_binary('-x_1+x_2', o)
        'x_2-x_1'

        >>> unary_minus_to_binary('x_1-x_2', o)
        'x_1-x_2'

        >>> unary_minus_to_binary('x_1+(-x_2+x_3)', o)
        'x_1+(x_3-x_2)'

        >>> unary_minus_to_binary('-tan(x_1-exp(x_2))', o)
        '(0-tan(x_1-exp(x_2)))'

        >>> unary_minus_to_binary('-x_2', o)
        '(0-x_2)'

        >>> unary_minus_to_binary('exp(-x_1)*log(x_2)', o)
        'exp((0-x_1))*log(x_2)'

        >>> unary_minus_to_binary('(c_1 + x_2)*(-c_2 + x_3)', o)
        '(c_1+x_2)*(x_3-c_2)'

        >>> unary_minus_to_binary('-(c_1 - x_1)^2', o)
        '(0-(c_1-x_1))^2'

        >>> unary_minus_to_binary('-(c_1 - x_2)*(x_1 + x_2)', o)
        '(0-(c_1-x_2))*(x_1+x_2)'

    """
    _temp = _find_unary("-", str(expr), operator_test)
    while "#" in _temp:
        _temp = _move_placeholder(_temp, operator_test)
    _temp = _find_unary("+", _temp, operator_test)
    _temp = __remove_character_from_string(_temp, "#")
    _temp = _find_unary("-", _temp, operator_test)
    while "#" in _temp:
        _temp = _replace_with_zero_minus(_temp, operator_test)
    return _temp


def _is_standard(s):
    pattern_v = r"^x_\d+$"
    pattern_c = r"^c_\d+$"
    return re.match(pattern_v, s) is not None or re.match(pattern_c, s) is not None


# TODO: make this more robust. Preferable only using function_test and operator_test. The tests
# for the constants and variables are still valid though (we standardize equations to use that)
def _infix_to_postfix(infix, function_test, operator_test):
    infix = "(" + infix + ")"
    n = len(infix)
    char_stack = []
    output = []
    i = 0
    while i < n:
        # Check if the character is alphabet or digit
        if infix[i].isdigit() and infix[i + 1] == "_":
            output.append(infix[i : i + 3][::-1])
            i += 2
        elif infix[i].isdigit() or infix[i] == "e":
            output.append(infix[i])
        elif infix[i] == "i" and infix[i + 1] == "p":
            output.append(infix[i : i + 2][::-1])
            i += 1

        # If the character is '(' push it in the stack
        elif infix[i] == "(":
            char_stack.append(infix[i])

        # If the character is ')' pop from the stack
        elif infix[i] == ")":
            while char_stack[-1] != "(":
                output.append(char_stack.pop())
            char_stack.pop()
        # Found an operator
        else:
            if (
                function_test(char_stack[-1])
                or operator_test(char_stack[-1])
                or char_stack[-1] in [")", "("]
            ):
                if infix[i] == "^":
                    while _get_priority(infix[i]) <= _get_priority(char_stack[-1]):
                        output.append(char_stack.pop())
                    char_stack.append(infix[i])
                elif infix[i] == "*" and i < n - 1 and infix[i + 1] == "*":
                    op = "**"
                    i += 1
                    while _get_priority(infix[i]) <= _get_priority(char_stack[-1]):
                        output.append(char_stack.pop())
                    char_stack.append(op)
                elif infix[i].isalpha():
                    fct = ""
                    while infix[i].isalpha() and i < n - 1:
                        fct += infix[i]
                        i += 1
                    i -= 1
                    while _get_priority(fct) < _get_priority(char_stack[-1]):
                        output.append(char_stack.pop())
                    char_stack.append(fct[::-1])
                else:  # + - * / ( )
                    while _get_priority(infix[i]) < _get_priority(char_stack[-1]):
                        output.append(char_stack.pop())
                    char_stack.append(infix[i])

        i += 1

    while len(char_stack) != 0:
        output.append(char_stack.pop())
    return output


def _get_priority(c):
    if c == "-" or c == "+":
        return 1
    elif c == "*" or c == "/":
        return 2
    elif c == "^" or c == "**":
        return 3
    elif c[0].isalpha():
        return 4
    return 0


def _find_unary(symbol, expression, operator_test):
    expression = expression.replace(" ", "")
    result = []
    prev_token = None

    for token in expression:
        if token == symbol:
            if prev_token is None or prev_token in ["("] or operator_test(prev_token):
                token = "#"
        result.append(token)
        prev_token = token
    return "".join(result)


def _move_placeholder(expression, operator_test):
    expression = expression.replace(" ", "")  # Remove any whitespaces
    i = 0
    has_to_close = False
    start = None
    end = None
    while i < len(expression):
        char = expression[i]
        if char == "(":
            has_to_close = True
        if char == "#":
            start = i
            open_brackets = 0
            # Move the expression after % to the end of the equation
            j = i + 1
            while (
                j < len(expression)
                and not operator_test(expression[j])
                or open_brackets >= 1
                or (j < len(expression) - 1 and expression[j] == "(")
            ):
                if j < len(expression) - 1 and expression[j] == "(":
                    open_brackets += 1
                if j < len(expression) and expression[j] == ")":
                    open_brackets -= 1
                    if open_brackets <= 0:
                        j += 1 + open_brackets
                        break
                j += 1

            end = j
            break
        i += 1
    if start is None or end is None:
        return
    if has_to_close:
        insert = __find_next_closing_parenthesis(expression, end)
    else:
        insert = len(expression)
    if end >= len(expression) - 1 or expression[end] == "+" or expression[end] == "-":
        replacement = "-" + expression[start:end][1:]
        new_expression = __insert_string_at_position(expression, replacement, insert)
        new_expression = __delete_chars_between_indexes(new_expression, start, end - 1)
        return new_expression
    else:
        modified_expression = expression[:start] + "-" + expression[start + 1 :]
        return modified_expression


def _replace_with_zero_minus(expression, operator_test):
    expression = expression.replace(" ", "")  # Remove any whitespaces
    i = 0
    start = None
    end = None
    while i < len(expression):
        char = expression[i]
        if char == "#":
            start = i
            open_brackets = 0
            # Move the expression after % to the end of the equation
            j = i + 1
            while (
                j < len(expression)
                and not operator_test(expression[j])
                or open_brackets >= 1
                or (j < len(expression) - 1 and expression[j] == "(")
            ):

                if j < len(expression) - 1 and expression[j] == "(":
                    open_brackets += 1
                if j < len(expression) and expression[j] == ")":
                    open_brackets -= 1
                    if open_brackets <= 0:
                        j += 1 + open_brackets
                        break
                j += 1
            end = j
            break
        i += 1
    if start is None or end is None:
        return
    replacement = "(0-" + expression[start:end][1:] + ")"
    new_expression = __insert_string_at_position(expression, replacement, start)
    new_expression = __delete_chars_between_indexes(
        new_expression, end + 3, end + 2 + (end - start)
    )
    return new_expression


def __delete_chars_between_indexes(input_string, i, j):
    if i < 0:
        i = 0
    if j >= len(input_string):
        j = len(input_string)

    return input_string[:i] + input_string[j + 1 :]


def __remove_character_from_string(input_string, character):
    return input_string.replace(character, "")


def __insert_string_at_position(original_string, string_to_insert, position):
    return original_string[:position] + string_to_insert + original_string[position:]


def __find_next_closing_parenthesis(input_string, j):
    for i in range(j, len(input_string)):
        if input_string[i] == ")":
            return i
    return -1  # Return -1 if closing parenthesis is not found after index j
