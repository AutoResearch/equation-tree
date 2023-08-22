from typing import Dict, List

from equation_tree.tree import EquationTree
from equation_tree.util.priors import normalized_dict


def get_frequencies(trees: List[EquationTree]):
    """
    Examples:
        >>> import numpy as np
        >>> import pprint
        >>> np.random.seed(42)
        >>> p = {
        ...     'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
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
         'operator_conditionals': {'+': {'features': {'constants': 0.5,
                                                      'variables': 0.5},
                                         'functions': {'cos': 0.5, 'sin': 0.5},
                                         'operators': {'+': 0.5, '-': 0.5}},
                                   '-': {'features': {'constants': 0.5,
                                                      'variables': 0.5},
                                         'functions': {'cos': 0.5, 'sin': 0.5},
                                         'operators': {'+': 0.5, '-': 0.5}}},
         'operators': {'+': 0.5, '-': 0.5},
         'structures': {'[0, 1, 1]': 0.3,
                        '[0, 1, 2, 3, 2, 3, 1]': 0.4,
                        '[0, 1, 2]': 0.3}}


        >>> tree_list = [EquationTree.from_prior(p, 4) for _ in range(100)]
        >>> len(tree_list)
        100
        >>> example_tree = tree_list[2]
        >>> example_tree.expr
        ['+', '-', 'sin', 'x_1', 'cos', 'x_2', 'c_1']
        >>> example_tree.sympy_expr
        c_1 + sin(x_1) - cos(x_2)
        >>> pprint.pprint(example_tree.info)
        {'depth': 3,
         'features': {'constants': 1, 'variables': 2},
         'function_conditionals': {'cos': {'features': {'constants': 0, 'variables': 1},
                                           'functions': {},
                                           'operators': {}},
                                   'sin': {'features': {'constants': 0, 'variables': 1},
                                           'functions': {},
                                           'operators': {}}},
         'functions': {'cos': 1, 'sin': 1},
         'max_depth': 7,
         'operator_conditionals': {'+': {'features': {'constants': 1, 'variables': 0},
                                         'functions': {},
                                         'operators': {'-': 1}},
                                   '-': {'features': {'constants': 0, 'variables': 0},
                                         'functions': {'cos': 1, 'sin': 1},
                                         'operators': {}}},
         'operators': {'+': 1, '-': 1},
         'structures': [0, 1, 2, 3, 2, 3, 1]}

        >>> frequencies = get_frequencies(tree_list)
        >>> pprint.pprint(frequencies)
        {'depth': {1: 0.32, 2: 0.27, 3: 0.41},
         'features': {'constants': 0.205607476635514, 'variables': 0.794392523364486},
         'function_conditionals': {'cos': {'features': {'constants': 0.0,
                                                        'variables': 1.0},
                                           'functions': {'cos': 0.5384615384615384,
                                                         'sin': 0.46153846153846156},
                                           'operators': {}},
                                   'sin': {'features': {'constants': 0.0,
                                                        'variables': 1.0},
                                           'functions': {'cos': 0.5, 'sin': 0.5},
                                           'operators': {}}},
         'functions': {'cos': 0.49264705882352944, 'sin': 0.5073529411764706},
         'max_depth': {3: 0.59, 7: 0.41},
         'operator_conditionals': {'+': {'features': {'constants': 0.4528301886792453,
                                                      'variables': 0.5471698113207547},
                                         'functions': {'cos': 0.5625, 'sin': 0.4375},
                                         'operators': {'+': 0.6086956521739131,
                                                       '-': 0.391304347826087}},
                                   '-': {'features': {'constants': 0.38461538461538464,
                                                      'variables': 0.6153846153846154},
                                         'functions': {'cos': 0.38235294117647056,
                                                       'sin': 0.6176470588235294},
                                         'operators': {'+': 0.5555555555555556,
                                                       '-': 0.4444444444444444}}},
         'operators': {'+': 0.543859649122807, '-': 0.45614035087719296},
         'structures': {'[0, 1, 1]': 0.32,
                        '[0, 1, 2, 3, 2, 3, 1]': 0.41,
                        '[0, 1, 2]': 0.27}}
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
        if "structures" in _info.keys():
            _update(structures, str(_info["structures"]))
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


def get_counts(trees: List[EquationTree]):
    """
    Examples:
        >>> import numpy as np
        >>> import pprint
        >>> np.random.seed(42)
        >>> p = {
        ...     'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
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
         'operator_conditionals': {'+': {'features': {'constants': 0.5,
                                                      'variables': 0.5},
                                         'functions': {'cos': 0.5, 'sin': 0.5},
                                         'operators': {'+': 0.5, '-': 0.5}},
                                   '-': {'features': {'constants': 0.5,
                                                      'variables': 0.5},
                                         'functions': {'cos': 0.5, 'sin': 0.5},
                                         'operators': {'+': 0.5, '-': 0.5}}},
         'operators': {'+': 0.5, '-': 0.5},
         'structures': {'[0, 1, 1]': 0.3,
                        '[0, 1, 2, 3, 2, 3, 1]': 0.4,
                        '[0, 1, 2]': 0.3}}


        >>> tree_list = [EquationTree.from_prior(p, 4) for _ in range(100)]
        >>> len(tree_list)
        100
        >>> example_tree = tree_list[2]
        >>> example_tree.expr
        ['+', '-', 'sin', 'x_1', 'cos', 'x_2', 'c_1']
        >>> example_tree.sympy_expr
        c_1 + sin(x_1) - cos(x_2)
        >>> pprint.pprint(example_tree.info)
        {'depth': 3,
         'features': {'constants': 1, 'variables': 2},
         'function_conditionals': {'cos': {'features': {'constants': 0, 'variables': 1},
                                           'functions': {},
                                           'operators': {}},
                                   'sin': {'features': {'constants': 0, 'variables': 1},
                                           'functions': {},
                                           'operators': {}}},
         'functions': {'cos': 1, 'sin': 1},
         'max_depth': 7,
         'operator_conditionals': {'+': {'features': {'constants': 1, 'variables': 0},
                                         'functions': {},
                                         'operators': {'-': 1}},
                                   '-': {'features': {'constants': 0, 'variables': 0},
                                         'functions': {'cos': 1, 'sin': 1},
                                         'operators': {}}},
         'operators': {'+': 1, '-': 1},
         'structures': [0, 1, 2, 3, 2, 3, 1]}

        >>> counts = get_counts(tree_list)
        >>> pprint.pprint(counts)
        {'depth': {1: 32, 2: 27, 3: 41},
         'features': {'constants': 44, 'variables': 170},
         'function_conditionals': {'cos': {'features': {'constants': 0,
                                                        'variables': 54},
                                           'functions': {'cos': 7, 'sin': 6},
                                           'operators': {}},
                                   'sin': {'features': {'constants': 0,
                                                        'variables': 55},
                                           'functions': {'cos': 7, 'sin': 7},
                                           'operators': {}}},
         'functions': {'cos': 67, 'sin': 69},
         'max_depth': {3: 59, 7: 41},
         'operator_conditionals': {'+': {'features': {'constants': 24, 'variables': 29},
                                         'functions': {'cos': 27, 'sin': 21},
                                         'operators': {'+': 14, '-': 9}},
                                   '-': {'features': {'constants': 20, 'variables': 32},
                                         'functions': {'cos': 13, 'sin': 21},
                                         'operators': {'+': 10, '-': 8}}},
         'operators': {'+': 62, '-': 52},
         'structures': {'[0, 1, 1]': 32, '[0, 1, 2, 3, 2, 3, 1]': 41, '[0, 1, 2]': 27}}
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
        if "structures" in _info.keys():
            _update(structures, str(_info["structures"]))
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


    info["max_depth"] = max_depth
    info["depth"] = depth
    info["structures"] = structures
    info["features"] = features
    info["functions"] = functions
    info["operators"] = operators
    info["function_conditionals"] = function_conditionals
    info["operator_conditionals"] = operator_conditionals
    return info


def _update(dct, key, val=1):
    if key in dct:
        dct[key] += val
    else:
        dct[key] = val
