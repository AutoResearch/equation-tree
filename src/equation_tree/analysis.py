from typing import Dict, List

from util.priors import normalized_dict

from equation_tree.tree import EquationTree


def get_frequencies(trees: List[EquationTree]):
    """
    Examples:
        >>> import numpy as np
        >>> import pprint
        >>> np.random.seed(42)
        >>> p = {'max_depth': 8,
        ...     'structure': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
        ...     'features': {'constants': .5, 'variables': .5},
        ...     'functions': {'sin': .5, 'cos': .5},
        ...     'operators': {'+': .5, '-': .5},
        ...     'function_conditionals': {
        ...                             'sin': {
        ...                                 'features': {'constants': 0., 'variables': 1.},
        ...                                 'functions': {'sin': .5, 'cos': .5},
        ...                                 'operators': {'+': .5, '-': .5}
        ...                             },
        ...                             'cos': {
        ...                                 'features': {'constants': 0., 'variables': 1.},
        ...                                 'functions': {'cos': .5, 'sin': .5},
        ...                                 'operators': {'+': .5, '-': .5}
        ...                             }
        ...                         },
        ...     'operator_conditionals': {
        ...                             '+': {
        ...                                 'features': {'constants': .5, 'variables': .5},
        ...                                 'functions': {'sin': .5, 'cos': .5},
        ...                                 'operators': {'+': .5, '-': .5}
        ...                             },
        ...                             '-': {
        ...                                 'features': {'constants': .5, 'variables': .5},
        ...                                 'functions': {'cos': .5, 'sin': .5},
        ...                                 'operators': {'+': .5, '-': .5}
        ...                             }
        ...                         },
        ... }
        >>> pprint.pprint(p)
        {'features': {'constants': 0.5, 'variables': 0.5},
         'function_conditionals': {'cos': {'features': {'constants': 0.0,
                                                        'variables': 1.0},
                                           'functions': {'cos': 0.5, 'sin': 0.5},
                                           'operators': {'+': 0.5, '-': 0.5}},
                                   'sin': {'features': {'constants': 0.0,
                                                        'variables': 1.0},
                                           'functions': {'cos': 0.5, 'sin': 0.5},
                                           'operators': {'+': 0.5, '-': 0.5}}},
         'functions': {'cos': 0.5, 'sin': 0.5},
         'max_depth': 8,
         'operator_conditionals': {'+': {'features': {'constants': 0.5,
                                                      'variables': 0.5},
                                         'functions': {'cos': 0.5, 'sin': 0.5},
                                         'operators': {'+': 0.5, '-': 0.5}},
                                   '-': {'features': {'constants': 0.5,
                                                      'variables': 0.5},
                                         'functions': {'cos': 0.5, 'sin': 0.5},
                                         'operators': {'+': 0.5, '-': 0.5}}},
         'operators': {'+': 0.5, '-': 0.5},
         'structure': {'[0, 1, 1]': 0.3,
                       '[0, 1, 2, 3, 2, 3, 1]': 0.4,
                       '[0, 1, 2]': 0.3}}


        >>> tree_list = [EquationTree.from_prior(p, 4) for _ in range(100)]
        >>> len(tree_list)
        100
        >>> example_tree = tree_list[2]
        >>> example_tree.expr
        ['+', '-', 'sin', 'x_1', 'sin', 'x_2', 'x_1']
        >>> example_tree.sympy_expr
        x_1 + sin(x_1) - sin(x_2)
        >>> pprint.pprint(example_tree.info)
        {'depth': 3,
         'features': {'constants': 0, 'variables': 3},
         'function_conditionals': {'sin': {'features': {'constants': 0, 'variables': 2},
                                           'functions': {},
                                           'operators': {}}},
         'functions': {'sin': 2},
         'max_depth': 7,
         'operator_conditionals': {'+': {'features': {'constants': 0, 'variables': 1},
                                         'functions': {},
                                         'operators': {'-': 1}},
                                   '-': {'features': {'constants': 0, 'variables': 0},
                                         'functions': {'sin': 2},
                                         'operators': {}}},
         'operators': {'+': 1, '-': 1},
         'structure': [0, 1, 2, 3, 2, 3, 1]}

        >>> frequencies = get_frequencies(tree_list)
        >>> pprint.pprint(frequencies)
        {'depth': {1: 0.29, 2: 0.32, 3: 0.39},
         'features': {'constants': 0.2318840579710145, 'variables': 0.7681159420289855},
         'function_conditionals': {'cos': {'features': {'constants': 0.0,
                                                        'variables': 1.0},
                                           'functions': {'cos': 0.5, 'sin': 0.5},
                                           'operators': {}},
                                   'sin': {'features': {'constants': 0.0,
                                                        'variables': 1.0},
                                           'functions': {'cos': 0.375, 'sin': 0.625},
                                           'operators': {}}},
         'functions': {'cos': 0.4154929577464789, 'sin': 0.5845070422535211},
         'max_depth': {3: 0.61, 7: 0.39},
         'operator_conditionals': {'+': {'features': {'constants': 0.46153846153846156,
                                                      'variables': 0.5384615384615384},
                                         'functions': {'cos': 0.3333333333333333,
                                                       'sin': 0.6666666666666666},
                                         'operators': {'+': 0.45454545454545453,
                                                       '-': 0.5454545454545454}},
                                   '-': {'features': {'constants': 0.5333333333333333,
                                                      'variables': 0.4666666666666667},
                                         'functions': {'cos': 0.40476190476190477,
                                                       'sin': 0.5952380952380952},
                                         'operators': {'+': 0.47058823529411764,
                                                       '-': 0.5294117647058824}}},
         'operators': {'+': 0.514018691588785, '-': 0.48598130841121495},
         'structures': {'[0, 1, 1]': 0.29,
                        '[0, 1, 2, 3, 2, 3, 1]': 0.39,
                        '[0, 1, 2]': 0.32}}
    """
    info: Dict = {}
    max_depth: Dict = {}
    depth: Dict = {}
    structures: Dict = {}
    features: Dict = {}
    functions: Dict = {}
    operators: Dict = {}
    function_conditionals: Dict = {}
    operator_conditionals: Dict = {}
    for t in trees:
        _info = t.info
        if "max_depth" in _info.keys():
            _update(max_depth, _info["max_depth"])
        if "depth" in _info.keys():
            _update(depth, _info["depth"])
        if "structure" in _info.keys():
            _update(structures, str(_info["structure"]))
        if "features" in _info.keys():
            for key, val in _info["features"].items():
                _update(features, key, val)
        if "functions" in _info.keys():
            for key, val in _info["functions"].items():
                _update(functions, key, val)
        if "operators" in _info.keys():
            for key, val in _info["operators"].items():
                _update(operators, key, val)
        if "function_conditionals" in _info.keys():
            fnc_con_dict = _info["function_conditionals"]
            for k, fnc_dict in fnc_con_dict.items():
                if k not in function_conditionals.keys():
                    function_conditionals[k] = {
                        "features": {},
                        "functions": {},
                        "operators": {},
                    }
                fnc_dct_features = fnc_dict["features"]
                fnc_fct_functions = fnc_dict["functions"]
                fnc_fct_operators = fnc_dict["operators"]
                for key, val in fnc_dct_features.items():
                    _update(function_conditionals[k]["features"], key, val)
                for key, val in fnc_fct_functions.items():
                    _update(function_conditionals[k]["functions"], key, val)
                for key, val in fnc_fct_operators.items():
                    _update(function_conditionals[k]["operators"], key, val)
        if "operator_conditionals" in _info.keys():
            op_con_dict = _info["operator_conditionals"]
            for k, op_dict in op_con_dict.items():
                if k not in operator_conditionals.keys():
                    operator_conditionals[k] = {
                        "features": {},
                        "functions": {},
                        "operators": {},
                    }
                op_dct_features = op_dict["features"]
                op_fct_functions = op_dict["functions"]
                op_fct_operators = op_dict["operators"]
                for key, val in op_dct_features.items():
                    _update(operator_conditionals[k]["features"], key, val)
                for key, val in op_fct_functions.items():
                    _update(operator_conditionals[k]["functions"], key, val)
                for key, val in op_fct_operators.items():
                    _update(operator_conditionals[k]["operators"], key, val)
    for k_c in function_conditionals:
        for k in ["features", "functions", "operators"]:
            if k in function_conditionals[k_c]:
                function_conditionals[k_c][k] = normalized_dict(
                    function_conditionals[k_c][k]
                )
    for k_c in operator_conditionals:
        for k in ["features", "functions", "operators"]:
            if k in operator_conditionals[k_c]:
                operator_conditionals[k_c][k] = normalized_dict(
                    operator_conditionals[k_c][k]
                )

    info["max_depth"] = normalized_dict(max_depth)
    info["depth"] = normalized_dict(depth)
    info["structures"] = normalized_dict(structures)
    info["features"] = normalized_dict(features)
    info["functions"] = normalized_dict(functions)
    info["operators"] = normalized_dict(operators)
    info["function_conditionals"] = function_conditionals
    info["operator_conditionals"] = operator_conditionals
    return info


def _update(dct, key, val=1):
    if key in dct:
        dct[key] += val
    else:
        dct[key] = val
