

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
    return _get_defined(prior, 'functions', 'function_conditionals')


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
    return _get_defined(prior, 'operators', 'operator_conditionals')


def _get_defined(prior, key, key_cond):
    _1 = [k for k in prior[key].keys()] if key in prior else []
    _2 = [k for k in prior[key_cond].keys()] if key_cond in prior else []
    _fnc_cnd_all = prior['function_conditionals'] if 'function_conditionals' in prior else {}
    _op_cnd_all = prior['operator_conditionals'] if 'operator_conditionals' in prior else {}
    _3 = []
    for k, val in _fnc_cnd_all.items():
        if key in val.keys():
            _3 += val[key]
    for k, val in _op_cnd_all.items():
        if key in val.keys():
            _3 += val[key]
    return list(set(_1 + _2 + _3))



