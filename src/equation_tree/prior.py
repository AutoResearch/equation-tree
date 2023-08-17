import copy

from equation_tree.src.sample_tree_structure import _gen_tree_structures
from equation_tree.util.priors import priors_from_space
from equation_tree.util.type_check import is_numeric

MIN = 0.001

DEFAULT_PRIOR_FUNCTIONS = priors_from_space(
    ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
)
DEFAULT_PRIOR_OPERATORS = priors_from_space(["+", "-", "*", "/", "^"])


def prior_from_space(space):
    """
    Uniform prior from a list of strings
    Examples:
        >>> import pprint
        >>> operator_space = ['+', '-', '*', '/', '^']
        >>> operator_prior = prior_from_space(operator_space)
        >>> pprint.pprint(operator_prior)
        {'*': 0.2, '+': 0.2, '-': 0.2, '/': 0.2, '^': 0.2}

        >>> function_space = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
        >>> function_prior = prior_from_space(function_space)
        >>> pprint.pprint(function_prior)
        {'abs': 0.14285714285714285,
         'cos': 0.14285714285714285,
         'exp': 0.14285714285714285,
         'log': 0.14285714285714285,
         'sin': 0.14285714285714285,
         'sqrt': 0.14285714285714285,
         'tan': 0.14285714285714285}
    """
    return priors_from_space(space)


def structure_prior_from_depth(depth):
    """
    Uniform structure prior for a given depth
    Examples:
        >>> import pprint
        >>> pprint.pprint(structure_prior_from_depth(3))
        {'[0, 1, 1]': 0.5, '[0, 1, 2]': 0.5}
        >>> pprint.pprint(structure_prior_from_depth(4))
        {'[0, 1, 1, 2]': 0.25,
         '[0, 1, 2, 1]': 0.25,
         '[0, 1, 2, 2]': 0.25,
         '[0, 1, 2, 3]': 0.25}

    """
    structures = _gen_tree_structures(depth)
    structures = [s for s in structures if len(s) == depth]
    return prior_from_space([str(s) for s in structures])


def structure_prior_from_max_depth(max_depth):
    """
    Uniform structure prior up to a given depth (from min depth 3)
    Examples:
        >>> import pprint
        >>> pprint.pprint(structure_prior_from_max_depth(3))
        {'[0, 1, 1]': 0.5, '[0, 1, 2]': 0.5}
        >>> pprint.pprint(structure_prior_from_max_depth(4))
        {'[0, 1, 1, 2]': 0.16666666666666666,
         '[0, 1, 1]': 0.16666666666666666,
         '[0, 1, 2, 1]': 0.16666666666666666,
         '[0, 1, 2, 2]': 0.16666666666666666,
         '[0, 1, 2, 3]': 0.16666666666666666,
         '[0, 1, 2]': 0.16666666666666666}
        >>> pprint.pprint(structure_prior_from_max_depth(5))
        {'[0, 1, 1, 2, 2]': 0.06666666666666667,
         '[0, 1, 1, 2, 3]': 0.06666666666666667,
         '[0, 1, 1, 2]': 0.06666666666666667,
         '[0, 1, 1]': 0.06666666666666667,
         '[0, 1, 2, 1, 2]': 0.06666666666666667,
         '[0, 1, 2, 1]': 0.06666666666666667,
         '[0, 1, 2, 2, 1]': 0.06666666666666667,
         '[0, 1, 2, 2, 3]': 0.06666666666666667,
         '[0, 1, 2, 2]': 0.06666666666666667,
         '[0, 1, 2, 3, 1]': 0.06666666666666667,
         '[0, 1, 2, 3, 2]': 0.06666666666666667,
         '[0, 1, 2, 3, 3]': 0.06666666666666667,
         '[0, 1, 2, 3, 4]': 0.06666666666666667,
         '[0, 1, 2, 3]': 0.06666666666666667,
         '[0, 1, 2]': 0.06666666666666667}
    """
    structures = _gen_tree_structures(max_depth)
    return prior_from_space([str(s) for s in structures])


def re_normalize(target, origin):
    """ """
    tmp_t = copy.deepcopy(target)
    tmp_o = copy.deepcopy(origin)

    def _rec_re_normalize(t, p):
        for k, val in t.items():
            if isinstance(val, dict) and isinstance(p[k], dict):
                _rec_re_normalize(t[k], p[k])
            elif not isinstance(val, dict):
                if not isinstance(p[k], dict) and val <= 0 < p[k]:
                    t[k] = MIN
                elif val <= 0:
                    t[k] = 0

    _rec_re_normalize(tmp_t, tmp_o)
    normalize(tmp_t)
    return tmp_t


def filter_keys(dict_a, dict_b):
    """
    Examples:
        >>> import pprint
        >>> a = {'a': {'a_1': 3, 'a_2': 2}, 'b': {'b_1': .1, 'b_2': .3}, 'c': {}}
        >>> b = {'a': {'a_1': 2, 'a_2': 4}, 'b': {'b_1': 3, 'c_1': 3}}
        >>> pprint.pprint(filter_keys(a, b))
        {'a': {'a_1': 3, 'a_2': 2}, 'b': {'b_1': 0.1}}
    """
    tmp_a = copy.deepcopy(dict_a)
    tmp_b = copy.deepcopy(dict_b)

    def _rec_f_key(a, b):
        for k in list(a.keys()):
            if k in b.keys():
                if isinstance(a[k], dict) and isinstance(b[k], dict):
                    _rec_f_key(a[k], b[k])
                else:
                    a[k] = a[k]
            else:
                del a[k]

    _rec_f_key(tmp_a, tmp_b)
    return tmp_a


def scalar_multiply(scalar, prior):
    """
    Examples:
        >>> import pprint
        >>> m = {'a': {'a_1': 3, 'a_2': 2}, 'b': {'b_1': .1, 'b_2': .3}}
        >>> s = 2
        >>> pprint.pprint(scalar_multiply(s, m))
        {'a': {'a_1': 6, 'a_2': 4}, 'b': {'b_1': 0.2, 'b_2': 0.6}}
    """
    tmp_p = copy.deepcopy(prior)

    def _rec_s_mult(s, p):
        for k, val in p.items():
            if isinstance(val, dict):
                _rec_s_mult(s, p[k])
            else:
                p[k] *= s

    _rec_s_mult(scalar, tmp_p)
    return tmp_p


def add(prior_a, prior_b):
    """
    >>> import pprint
    >>> a = {'a': {'a_1': 3, 'a_2': 2}, 'b': {'b_1': .1, 'b_2': .3}}
    >>> b = {'a': {'a_1': 2, 'a_2': 4}, 'c': {'b_1': 3, 'b_2': 3}}
    >>> pprint.pprint(add(a, b))
    {'a': {'a_1': 5, 'a_2': 6},
     'b': {'b_1': 0.1, 'b_2': 0.3},
     'c': {'b_1': 3, 'b_2': 3}}

    """
    tmp_a = copy.deepcopy(prior_a)
    tmp_b = copy.deepcopy(prior_b)

    def _rec_add(a, b):
        for k, val in a.items():
            if k in b.keys():
                if isinstance(val, dict) and isinstance(b[k], dict):
                    _rec_add(a[k], b[k])
                elif not isinstance(val, dict) and not isinstance(b[k], dict):
                    a[k] += b[k]

    _rec_add(tmp_a, tmp_b)
    return append(tmp_a, tmp_b)


def subtract(minuend, subtrahend):
    """
    Minuend - Subtrahend
    Examples:
        >>> import pprint
        >>> m = {'a': {'a_1': 3, 'a_2': 2}, 'b': {'b_1': .1, 'b_2': .3}}
        >>> s = {'a': {'a_1': 2, 'a_2': 4}}
        >>> pprint.pprint(subtract(m, s))
        {'a': {'a_1': 1, 'a_2': -2}, 'b': {'b_1': 0.0, 'b_2': 0.0}}

    """
    tmp_m = copy.deepcopy(minuend)
    tmp_s = copy.deepcopy(subtrahend)

    def _rec_sub(m, s):
        for k, val in m.items():
            if k in s.keys():
                if isinstance(val, dict) and isinstance(s[k], dict):
                    _rec_sub(val, s[k])
                elif (
                    not isinstance(val, dict)
                    and is_numeric(val)
                    and not isinstance(s[k], dict)
                    and is_numeric(s[k])
                ):
                    m[k] -= s[k]
                else:
                    if not isinstance(val, dict):
                        m[k] = 0.0
                    else:
                        _set_zero(m[k])
            else:
                if not isinstance(val, dict):
                    m[k] = 0.0
                else:
                    _set_zero(m[k])

    _rec_sub(tmp_m, tmp_s)
    return tmp_m


def multiply(a, b):
    """
    Examples:
        >>> import pprint
        >>> prior_a = {'a':
        ...                 {'a_1':  .2, 'a_2': .8},
        ...             'b': {
        ...                 'b_1' : {'b_1_1': .25, 'b_1_2': .5, 'b_1_3': .25}
        ...             }
        ...         }
        >>> prior_b = {'a':
        ...                 {'a_1':  4, 'a_2': 1},
        ...             'b': {
        ...                 'b_1' : {'b_1_1': 4 , 'b_1_2': 1, 'b_1_3': 4}
        ...             },
        ...             'c': {'c_1': .9, 'c_2': .1}
        ...         }
        >>> product = multiply(prior_a, prior_b)
        >>> pprint.pprint(product)
        {'a': {'a_1': 0.8, 'a_2': 0.8},
         'b': {'b_1': {'b_1_1': 1.0, 'b_1_2': 0.5, 'b_1_3': 1.0}},
         'c': {'c_1': 0.9, 'c_2': 0.1}}
        >>> normalize(product)
        >>> pprint.pprint(product)
        {'a': {'a_1': 0.5, 'a_2': 0.5},
         'b': {'b_1': {'b_1_1': 0.4, 'b_1_2': 0.2, 'b_1_3': 0.4}},
         'c': {'c_1': 0.9, 'c_2': 0.1}}
    """
    tmp_a = copy.deepcopy(a)
    tmp_b = copy.deepcopy(b)

    def _multiply(_a, _b):
        for k, val in _a.items():
            if k in _b.keys():
                if isinstance(val, dict) and isinstance(_b[k], dict):
                    _multiply(val, _b[k])
                elif not isinstance(val, dict) and not isinstance(_b[k], dict):
                    res = val * _b[k]
                    _a[k] = res

    _multiply(tmp_a, tmp_b)
    tmp_a = append(tmp_a, tmp_b)

    return tmp_a


def append(a, b):
    tmp_a = a.copy()
    tmp_b = b.copy()
    _rec_append(tmp_a, tmp_b)
    return tmp_a


def _rec_append(a, b):
    """
    Recursivly add all dicts from a to b and vice versa
    Examples:
        >>> import pprint
        >>> test_a = {'a': {'b': 1, 'c': 2}, 'd': {'b': {'a': 1.}}}
        >>> test_b = {'a': {'b': 2, 'c': 3}, 'e': {'l': 1, 'f': 1}, 'd': {'b': {}, 'e': {}}}
        >>> _rec_append(test_a, test_b)
        >>> pprint.pprint(test_a)
        {'a': {'b': 1, 'c': 2}, 'd': {'b': {'a': 1.0}, 'e': {}}, 'e': {'f': 1, 'l': 1}}
        >>> pprint.pprint(test_b)
        {'a': {'b': 2, 'c': 3}, 'd': {'b': {'a': 1.0}, 'e': {}}, 'e': {'f': 1, 'l': 1}}
    """
    for k, val in a.items():
        if k in b.keys():
            if isinstance(val, dict) and isinstance(b[k], dict):
                _rec_append(val, b[k])
        else:
            b[k] = a[k]
    for k, val in b.items():
        if k in a.keys():
            if isinstance(val, dict) and isinstance(a[k], dict):
                _rec_append(a[k], val)
        else:
            a[k] = b[k]


def default(prior):
    tmp = copy.deepcopy(prior)
    _set_default(tmp)
    return tmp


def normalize(prior):
    for k, val in prior.items():
        if not isinstance(val, dict):
            d = sum(prior.values())
            for key in prior.keys():
                prior[key] /= d
        else:
            normalize(prior[k])


def _set_default(prior):
    """
    Examples:
        >>> import pprint
        >>> test_prior = {
        ...     'functions': {'sin': 2},
        ...     'function_conditionals': {'sin': {'features': {'constants': 0, 'variables': 1.},
        ...                                   'functions': {},
        ...                                       'operators': {}}},
        ...     'operator_conditionals': {'+': {'features': {'constants': .3, 'variables': .7},
        ...                                     'functions': {'cos': 1.},
        ...                                     'operators': {'-': .3}},
        ...                               '-': {'features': {'constants': .8, 'variables': .2},
        ...                                     'functions': {'sin': 2, 'abs': 1},
        ...                                     'operators': {}}},
        ...     'operators': {'+': 1, '-': 1},
        ...     'structures': {'[0, 1, 2, 3, 2, 3, 1]': 1}}
        >>> _set_default(test_prior)
        >>> pprint.pprint(test_prior)
        {'function_conditionals': {'sin': {'features': {'constants': 0.5,
                                                        'variables': 0.5},
                                           'functions': {},
                                           'operators': {}}},
         'functions': {'sin': 1.0},
         'operator_conditionals': {'+': {'features': {'constants': 0.5,
                                                      'variables': 0.5},
                                         'functions': {'cos': 1.0},
                                         'operators': {'-': 1.0}},
                                   '-': {'features': {'constants': 0.5,
                                                      'variables': 0.5},
                                         'functions': {'abs': 0.5, 'sin': 0.5},
                                         'operators': {}}},
         'operators': {'+': 0.5, '-': 0.5},
         'structures': {'[0, 1, 2, 3, 2, 3, 1]': 1.0}}

    """
    for k, val in prior.items():
        if not isinstance(val, dict):
            prior[k] = 1.0 / len(prior.keys())
        else:
            _set_default(val)


def _set_zero(prior):
    if not isinstance(prior, dict):
        return
    for k, val in prior.items():
        if not isinstance(val, dict):
            prior[k] = 0.0
        else:
            _set_default(prior[k])


def get_defined_functions(prior):
    """
    Examples:
        >>> test_prior = {
        ...     'functions': {'sin': 2},
        ...     'function_conditionals': {'sin': {'features': {'constants': 0, 'variables': 2},
        ...                                   'functions': {},
        ...                                       'operators': {}}},
        ...     'operator_conditionals': {'+': {'features': {'constants': 0, 'variables': 1},
        ...                                     'functions': {'cos': 2},
        ...                                     'operators': {'-': 1}},
        ...                               '-': {'features': {'constants': 0, 'variables': 0},
        ...                                     'functions': {'sin': 2, 'abs': 1},
        ...                                     'operators': {}}},
        ...     'operators': {'+': 1, '-': 1},
        ...     'structure': [0, 1, 2, 3, 2, 3, 1]}
        >>> set(get_defined_functions(test_prior)) == set(['sin', 'abs', 'cos'])
        True
    """
    return _get_defined(prior, "functions", "function_conditionals")


def get_defined_operators(prior):
    """
    Examples:
        >>> test_prior = {
        ...     'function_conditionals': {'sin': {'features': {'constants': 0, 'variables': 2},
        ...                                   'functions': {},
        ...                                       'operators': {}}},
        ...     'functions': {'sin': 2},
        ...     'operator_conditionals': {'+': {'features': {'constants': 0, 'variables': 1},
        ...                                     'functions': {'cos': 2},
        ...                                     'operators': {'**': 1}},
        ...                               '/': {'features': {'constants': 0, 'variables': 0},
        ...                                     'functions': {'sin': 2},
        ...                                     'operators': {}}},
        ...     'operators': {'+': 1, '-': 1},
        ...     'structure': [0, 1, 2, 3, 2, 3, 1]}
        >>> set(get_defined_operators(test_prior)) == set(['+', '/', '**', '-'])
        True
    """
    return _get_defined(prior, "operators", "operator_conditionals")


def _get_defined(prior, key, key_cond):
    _1 = [k for k in prior[key].keys()] if key in prior else []
    _2 = [k for k in prior[key_cond].keys()] if key_cond in prior else []
    _fnc_cnd_all = (
        prior["function_conditionals"] if "function_conditionals" in prior else {}
    )
    _op_cnd_all = (
        prior["operator_conditionals"] if "operator_conditionals" in prior else {}
    )
    _3 = []
    for k, val in _fnc_cnd_all.items():
        if key in val.keys():
            _3 += val[key]
    for k, val in _op_cnd_all.items():
        if key in val.keys():
            _3 += val[key]
    return list(set(_1 + _2 + _3))
