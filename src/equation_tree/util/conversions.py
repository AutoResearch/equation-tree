import re

from sympy import symbols

from equation_tree.util.type_check import is_numeric, is_variable_formatted, is_constant_formatted

SYMPY_TO_ARITHMETIC = {
    "Add": "+",
    "Mul": "*",
    "Pow": "^",
}

CONVERSIONS_FUNC_OP_CONST = {"squared": "**2", "cubed": "**3"}

CONVERSION_OP_CONST_FUNC = {
    "**2": "squared",
    "**3": "cubed",
    "^2": "squared",
    "^3": "cubed",
}


def _sympy_fun_to_aritmethic(fun, opertor_test):
    if fun in SYMPY_TO_ARITHMETIC.keys():
        tmp = SYMPY_TO_ARITHMETIC[fun]
        if opertor_test("**") and tmp == "^":
            return "**"
        return tmp.lower()
    return fun.lower()


def prefix_to_infix(
    prefix, function_test=lambda _: False, operator_test=lambda _: False
):
    """
    Transforms prefix notation to infix notation

    Example:
        >>> is_function = lambda x: x in ['sin', 'cos']
        >>> is_operator = lambda x : x in ['+', '-', '*', 'max', '**']
        >>> prefix_to_infix(['-', 'x_1', 'x_2'], is_function, is_operator)
        '(x_1-x_2)'

        >>> prefix_to_infix(
        ...     ['*', 'x', 'cos', '+', 'y', 'z'], is_function, is_operator)
        '(x*cos((y+z)))'

        >>> prefix_to_infix(['max', 'x_1', 'x_2'], is_function, is_operator)
        'max(x_1,x_2)'

        >>> prefix_to_infix(['**', 'x_1', 'x_2'], is_function, is_operator)
        '(x_1**x_2)'

    """
    stack = []
    for i in range(len(prefix) - 1, -1, -1):
        if function_test(prefix[i]):
            # symbol in unary operator
            stack.append(prefix[i] + "(" + stack.pop() + ")")
        elif (operator_test(prefix[i]) or prefix[i] == "**") and prefix[i] in [
            "+",
            "-",
            "/",
            "^",
            "*",
            "**",
        ]:
            # symbol is binary operator
            str = "(" + stack.pop() + prefix[i] + stack.pop() + ")"
            stack.append(str)
        elif operator_test(prefix[i]):
            str = prefix[i] + "(" + stack.pop() + "," + stack.pop() + ")"
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
        >>> is_operator = lambda x : x in ['+', '-', '*', '/']
        >>> infix_to_prefix('x_2-x_1', is_function, is_operator)
        ['-', 'x_2', 'x_1']

        >>> infix_to_prefix('x_1-(x_2+x_4)', is_function, is_operator)
        ['-', 'x_1', '+', 'x_2', 'x_4']

        >>> infix_to_prefix('x_1*cos(c_1+x_2)', is_function, is_operator)
        ['*', 'x_1', 'cos', '+', 'c_1', 'x_2']

        >>> is_function = lambda x: x in ['sin', 'cos', 'e']
        >>> is_operator = lambda x: x in ['+', '-', '*', '^', 'max', '**', '/']
        >>> infix_to_prefix('x_1 + max(x_2, x_3)', is_function, is_operator)
        ['+', 'x_1', 'max', 'x_2', 'x_3']

        >>> infix_to_prefix('x_1-(x_2/(x_3-x_4))',is_function, is_operator)
        ['-', 'x_1', '/', 'x_2', '-', 'x_3', 'x_4']

        >>> infix_to_prefix('x_1^(sin(x_2)/x_3)', is_function, is_operator)
        ['^', 'x_1', '/', 'sin', 'x_2', 'x_3']

        >>> infix_to_prefix('sin(x_1)-x_2', is_function, is_operator)
        ['-', 'sin', 'x_1', 'x_2']
    """

    # n = len(infix)

    # infix = list(infix[::-1].lower())
    #
    # for i in range(n):
    #     if infix[i] == "(":
    #         infix[i] = ")"
    #     elif infix[i] == ")":
    #         infix[i] = "("
    #
    # infix = "".join(infix)
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
            if is_variable_formatted(str(node)):
                new_symbol = symbols(str(node))
                variables[str(node)] = new_symbol
                return new_symbol
            if not str(node) in variables.keys() or str(node):
                variable_count += 1
                new_symbol = symbols(f"x_{variable_count}")
                variables[str(node)] = new_symbol
            else:
                new_symbol = variables[str(node)]
            return new_symbol
        elif constant_test(str(node)) and not is_numeric(str(node)):
            if is_constant_formatted(str(node)):
                new_symbol = symbols(str(node))
                constants[str(node)] = new_symbol
                return new_symbol
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
        >>> o_ = lambda x : x in ['+', '-', '*', '/', '**']
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

        >>> unary_minus_to_binary('x_1**2 + x_2', o_)
        'x_1**2+x_2'

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


def _is_symbol(chars):
    """
    Examples:
        >>> _is_symbol('(')
        False
        >>> _is_symbol(')')
        False
    """
    return (
        (len(chars) == 3 and chars[-1].isdigit() and chars[-2] == "_")
        or (is_numeric(chars) or chars == "e")
        or (chars == "pi")
    )


def _is_token(chars, function_test, operator_test):
    return (
        _is_symbol(chars)
        or (chars == "(" or chars == ")" or chars == ",")
        or (function_test(chars.lower()))
        or (operator_test(chars.lower()) or chars == "**" or chars == "^")
    )


def _tokenize_infix(infix, function_test, operator_test):
    """
    Examples:
        >>> is_function = lambda a : a in ['sin', 'asin', 'abs']
        >>> is_operator = lambda a : a in ['+', '-', 'max', '*']
        >>> _tokenize_infix('x_1+asin(x_2)', is_function, is_operator)
        ['x_1', '+', 'asin', '(', 'x_2', ')']

        >>> _tokenize_infix('x_1-asin(max(x_2, sin(x_3)))', is_function, is_operator)
        ['x_1', '-', 'asin', '(', 'max', '(', 'x_2', ',', 'sin', '(', 'x_3', ')', ')', ')']

        >>> _tokenize_infix('2*x_1', is_function, is_operator)
        ['2', '*', 'x_1']


    """

    res = [""] * len(infix)
    for lng in reversed(range(len(infix))):
        start = 0

        while start + lng < len(infix):
            token = infix[start : start + lng + 1]
            if _is_token(token, function_test, operator_test):
                res[start] = token
                infix = infix[:start] + " " * len(token) + infix[start + lng + 1 :]
            start += 1
    res = [r for r in res if r != ""]
    return res


def _infix_to_postfix(infix, function_test, operator_test):
    """
    Example:
        >>> is_function = lambda x: x in ['sin', 'cos']
        >>> is_operator = lambda x : x in ['+', '-', '*', '/']
        >>> _infix_to_postfix('x_2-x_1', is_function, is_operator)[::-1]
        ['-', 'x_2', 'x_1']

        >>> _infix_to_postfix('x_1-(x_2+x_4)', is_function, is_operator)[::-1]
        ['-', 'x_1', '+', 'x_2', 'x_4']

        >>> _infix_to_postfix('x_1*cos(c_1+x_2)', is_function, is_operator)[::-1]
        ['*', 'x_1', 'cos', '+', 'c_1', 'x_2']

        >>> is_function = lambda x: x in ['sin', 'cos', 'e']
        >>> is_operator = lambda x: x in ['+', '-', '*', '^', 'min', 'max', '**', '/']
        >>> _infix_to_postfix('x_1 + max(x_2, x_3)', is_function, is_operator)[::-1]
        ['+', 'x_1', 'max', 'x_2', 'x_3']

        >>> _infix_to_postfix('x_1 + max(min(x_2 + x_4, x_5), x_3)', is_function, is_operator)[::-1]
        ['+', 'x_1', 'max', 'min', '+', 'x_2', 'x_4', 'x_5', 'x_3']

        >>> _infix_to_postfix('x_1-(x_2/(x_3-x_4))',is_function, is_operator)[::-1]
        ['-', 'x_1', '/', 'x_2', '-', 'x_3', 'x_4']

        >>> _infix_to_postfix('x_1^(sin(x_2)/x_3)', is_function, is_operator)[::-1]
        ['^', 'x_1', '/', 'sin', 'x_2', 'x_3']

        >>> _infix_to_postfix('sin(x_1)-x_2', is_function, is_operator)[::-1]
        ['-', 'sin', 'x_1', 'x_2']

        >>> _infix_to_postfix('x_1**x_2', is_function, is_operator)[::-1]
        ['**', 'x_1', 'x_2']
    """
    infix = _tokenize_infix(infix, function_test, operator_test)
    infix = [el.lower() for el in infix]
    infix = infix[::-1]
    for i in range(len(infix)):
        if infix[i] == "(":
            infix[i] = ")"
        elif infix[i] == ")":
            infix[i] = "("
    infix = ["("] + infix + [")"]

    n = len(infix)
    char_stack = []
    output = []
    i = 0

    while i < n:
        token = infix[i]
        if _is_symbol(token):
            output.append(token)
        elif token == "(":
            char_stack.append(token)
        elif token == ")":
            while char_stack[-1] != "(":
                output.append(char_stack.pop())
            char_stack.pop()
        else:
            if token == "^" or token == "**":
                while _get_priority(token) <= _get_priority(char_stack[-1]):
                    output.append(char_stack.pop())
                char_stack.append(token)
            elif function_test(token):
                while _get_priority(token) < _get_priority(char_stack[-1]):
                    output.append(char_stack.pop())
                char_stack.append(token)
            elif operator_test(token) and token not in ["+", "-", "*", "/", "^", "**"]:
                output.append(token)
            elif token != ",":
                while _get_priority(token) < _get_priority(char_stack[-1]):
                    output.append(char_stack.pop())
                char_stack.append(token)
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


def _remove_unnecessary_parentheses(expr):
    """
    Examples:
        >>> _remove_unnecessary_parentheses('((0-.06)*x_1+x_2)/(squared((x_1))+1)')
        '((0-.06)*x_1+x_2)/(squared(x_1)+1)'
        >>> _remove_unnecessary_parentheses('x_1*cubed((x_1*3-squared(c_1/x_1*2)-x_3))-squared(x)')
        'x_1*cubed(x_1*3-squared(c_1/x_1*2)-x_3)-squared(x)'
    """
    return expr
    stack = []
    to_remove = set()

    for i, char in enumerate(expr):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            if i > 0 and expr[i - 1] == ')' and expr[stack[-1]] == '(':
                to_remove.add(stack[-1])
                to_remove.add(i)
            stack.pop()

    return ''.join([char for i, char in enumerate(expr) if i not in to_remove])



def _op_const_func_rec(expr, el, key):
    expr = re.sub(r"(\b\w+\b)(\*\*)(\d)", r"(\1)**\3", expr)
    match = re.search(re.escape(el), expr)
    if not match:
        return expr

    open_count = 0
    start_idx = match.start() - 1
    end_idx = match.start()

    # Check if there is already a parentheses around the term
    if start_idx >= 0 and expr[start_idx] == "(" and expr[match.end()] == ")":
        return expr

    while start_idx >= 0:
        if expr[start_idx] == ")":
            open_count += 1
        elif expr[start_idx] == "(":
            open_count -= 1
            if open_count == 0:
                break
        elif (
            open_count == 0
            and not expr[start_idx].isalnum()
            and expr[start_idx] not in ["_", "."]
        ):
            start_idx += 1
            break
        start_idx -= 1

    return (
        expr[:start_idx]
        + key
        + "("
        + _op_const_func_rec(expr[start_idx:end_idx], el, key)
        + ")"
        + _op_const_func_rec(expr[match.end() :], el, key)
    )


def op_const_to_func(expr):
    """
    Known operators with constants to functions. For exampl, e**2->squared
    Examples:
        >>> op_const_to_func('x_1*(x_1*3-(c_1/x_1*2)**2-x_3)**3-(x)**2')
        'x_1*cubed(x_1*3-squared(c_1/x_1*2)-x_3)-squared(x)'
        >>> op_const_to_func('(x_1)**3')
        'cubed(x_1)'
        >>> op_const_to_func('x**2')
        'squared(x)'
        >>> op_const_to_func('c_1**3')
        'cubed(c_1)'
        >>> op_const_to_func('(x_2**2)')
        'squared(x_2)'
    """

    for key, el in CONVERSION_OP_CONST_FUNC.items():
        expr = _op_const_func_rec(expr, key, el)
    expr = _remove_unnecessary_parentheses(expr)

    return expr


def _func_op_const_rec(expr, key, el):
    match = re.search(rf"{key}\(", expr)

    if not match:
        return expr

    open_count = 0
    start_idx = match.end()
    for idx, char in enumerate(expr[start_idx:], start=start_idx):
        if char == "(":
            open_count += 1
        elif char == ")":
            if open_count == 0:
                return (
                    expr[: match.start()]
                    + "("
                    + _func_op_const_rec(expr[start_idx:idx], key, el)
                    + ")"
                    + el
                    + _func_op_const_rec(expr[idx + 1 :], key, el)
                )
            open_count -= 1

    raise ValueError("Unmatched parentheses in expression.")


def func_to_op_const(expr):
    """
    Examples:
        >>> e_1 = 'x_1*cubed(x_1*3-squared(c_1/x_1*2)-x_3)-squared(x)'
        >>> func_to_op_const(e_1)
        'x_1*(x_1*3-(c_1/x_1*2)**2-x_3)**3-(x)**2'
        >>> func_to_op_const('cubed(x_1)')
        '(x_1)**3'
    """
    for key, el in CONVERSIONS_FUNC_OP_CONST.items():
        expr = _func_op_const_rec(expr, key, el)
    return expr
