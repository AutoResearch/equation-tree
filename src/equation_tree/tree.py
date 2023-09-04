import copy
import warnings
from typing import Callable, Dict, List, Union, Optional
import yaml
import os

import numpy as np
import pandas as pd
from sympy import simplify, symbols, sympify
import sympy

from equation_tree.src.tree_node import (
    NodeKind,
    TreeNode,
    node_from_prefix,
    sample_tree,
    sample_tree_full,
)
from equation_tree.util.conversions import (
    infix_to_prefix,
    prefix_to_infix,
    standardize_sympy,
    unary_minus_to_binary,
)
from equation_tree.util.type_check import check_functions, is_known_constant, is_numeric

UnaryOperator = Callable[[Union[int, float]], Union[int, float]]
BinaryOperator = Callable[[Union[int, float], Union[int, float]], Union[int, float]]

OPERATORS: Dict[str, BinaryOperator] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "^": lambda a, b: a ** b,
    "**": lambda a, b: a ** b,
}

FUNCTIONS: Dict[str, UnaryOperator] = {
    "sin": lambda a: np.sin(a),
    "cos": lambda a: np.cos(a),
    "tan": lambda a: np.tan(a),
    "exp": lambda a: np.exp(a),
    "log": lambda a: np.log(a),
    "sqrt": lambda a: np.sqrt(a),
    "abs": lambda a: np.abs(a),
    "arg": lambda a: np.pi * a / 180
}


class EquationTree:
    """
    Equation tree that represents an equation as binary tree.
    """

    def __init__(self, node: TreeNode):
        """
        Initializes a tree from a TreeNode

        Examples:
            # We can inititlize from a single node
            >>> node_root = TreeNode(kind=NodeKind.VARIABLE, attribute="x")
            >>> equation_tree = EquationTree(node_root)
            >>> equation_tree.expr
            ['x']
            >>> equation_tree.structure
            [0]
            >>> equation_tree.variables
            ['x']

            # Or from a node with children
            >>> node_left = TreeNode(kind=NodeKind.VARIABLE, attribute="x")
            >>> node_right = TreeNode(kind=NodeKind.CONSTANT, attribute="c")
            >>> node_root = TreeNode(kind=NodeKind.OPERATOR, attribute="+", \
                            left=node_left, right=node_right)
            >>> equation_tree = EquationTree(node_root)
            >>> equation_tree.expr
            ['+', 'x', 'c']
            >>> equation_tree.structure
            [0, 1, 1]
            >>> equation_tree.variables
            ['x']
            >>> equation_tree.constants
            ['c']
            >>> equation_tree.operators
            ['+']

            # We can first sample a node and children and initialize from that
            >>> np.random.seed(42)
            >>> max_depth = 12
            >>> structure_priors = {'[0, 1, 2, 1, 2, 3]': .5, '[0, 1, 2, 2, 1, 2, 3]': .5}
            >>> feature_priors = {"x_1": 0.5, "c_1": 0.5}
            >>> function_priors = {"sin": 0.5, "cos": 0.5}
            >>> operator_priors = {"+": 0.5, "-": 0.5}
            >>> node_root = sample_tree(feature_priors, function_priors,
            ...     operator_priors, structure_priors)
            >>> equation_tree = EquationTree(node_root)
            >>> equation_tree.expr
            ['-', 'cos', 'c_1', 'sin', 'sin', 'x_1']
            >>> equation_tree.structure
            [0, 1, 2, 1, 2, 3]
            >>> equation_tree.variables
            ['x_1']
            >>> equation_tree.n_variables
            1
            >>> equation_tree.n_variables_unique
            1
            >>> equation_tree.constants
            ['c_1']
            >>> equation_tree.n_constants
            1
            >>> equation_tree.n_constants_unique
            1
            >>> equation_tree.n_leafs
            2
            >>> equation_tree.operators
            ['-']
            >>> equation_tree.functions
            ['cos', 'sin', 'sin']

            # First we create test functions that test weather an attribute is a variable,
            # a constant, a function, or an operater
            >>> is_variable = lambda x : x in ['x', 'y', 'z']
            >>> is_constant = lambda x : x in ['0', '1', '2']
            >>> is_function = lambda x : x in ['sin', 'cos']
            >>> is_operator = lambda x: x in ['+', '-', '*', '/']

            # here we use the prefix notation
            >>> prefix_notation = ['+', '-', 'x', '1', '*', 'sin', 'y', 'cos', 'z']

            # then we create the node root
            >>> node_root = node_from_prefix(
            ...     prefix_notation=prefix_notation,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     function_test=is_function,
            ...     operator_test=is_operator
            ...     )

            # and initialize the tree
            >>> equation_tree = EquationTree(node_root)
            >>> equation_tree.structure
            [0, 1, 2, 2, 1, 2, 3, 2, 3]
            >>> equation_tree.variables
            ['x', 'y', 'z']
            >>> equation_tree.constants
            ['1']
            >>> equation_tree.operators
            ['+', '-', '*']
            >>> equation_tree.functions
            ['sin', 'cos']

            # The tree expression is the same as the prefix notation
            >>> equation_tree.expr == prefix_notation
            True
        """

        self.root: Union[TreeNode, None] = node

        self.structure: List[int] = []

        self.expr: List[str] = list()

        self.variables: List[str] = list()
        self.functions: List[str] = list()
        self.operators: List[str] = list()
        self.constants: List[str] = list()

        self.evaluation = None

        self._build()

    @classmethod
    def from_prefix(
            cls,
            prefix_notation: List[str],
            function_test: Callable = lambda _: False,
            operator_test: Callable = lambda _: False,
            variable_test: Callable = lambda _: False,
            constant_test: Callable = lambda _: False,
    ):
        """
        Instantiate a tree from prefix notation

        Args:
            prefix_notation: The equation in prefix notation
            function_test: A function that tests if the attribute is a function
            operator_test: A function that tests if the attribute is an operator
            variable_test: A function that tests if the attribute is a variable
            constant_test: A function that tests if the attribute is a constant

        Example:
            >>> is_variable = lambda x : x in ['x', 'y', 'z']
            >>> is_constant = lambda x : x in ['0', '1', '2']
            >>> is_function = lambda x : x in ['sin', 'cos']
            >>> is_operator = lambda x: x in ['+', '-', '*', '/']
            >>> prefix = ['+', '-', 'x', '1', '*', 'sin', 'y', 'cos', 'z']

            # then we create the node root
            >>> equation_tree = EquationTree.from_prefix(
            ...     prefix_notation=prefix,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     function_test=is_function,
            ...     operator_test=is_operator
            ...     )

            # and initialize the tree
            >>> equation_tree.structure
            [0, 1, 2, 2, 1, 2, 3, 2, 3]
            >>> equation_tree.variables
            ['x', 'y', 'z']
            >>> equation_tree.constants
            ['1']
            >>> equation_tree.operators
            ['+', '-', '*']
            >>> equation_tree.functions
            ['sin', 'cos']

            # The tree expression is the same as the prefix notation
            >>> equation_tree.expr == prefix
            True


        """
        root = node_from_prefix(
            prefix_notation, function_test, operator_test, variable_test, constant_test
        )
        return cls(root)

    @classmethod
    def from_full_prior(cls, prior):
        root = sample_tree_full(prior)
        return cls(root)

    @classmethod
    def from_prior(cls, prior: Dict, max_variables_unique: int):
        """
        Initiate a tree from a prior

        Args:
            prior: The priors in dictionary form
            max_variables_unique: The maximum number of unique variables (a tree can have less then
                this number)

        Examples:
            >>> np.random.seed(42)

            # We can set priors for features, functions, operators
            # and also conditionals based the parent
            >>> p = {
            ...     'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
            ...     'features': {'constants': .2, 'variables': .8},
            ...     'functions': {'sin': .5, 'cos': .5},
            ...     'operators': {'+': 1., '-': .0},
            ...     'function_conditionals': {
            ...                             'sin': {
            ...                                 'features': {'constants': 0., 'variables': 1.},
            ...                                 'functions': {'sin': 0., 'cos': 1.},
            ...                                 'operators': {'+': 0., '-': 1.}
            ...                             },
            ...                             'cos': {
            ...                                 'features': {'constants': 0., 'variables': 1.},
            ...                                 'functions': {'cos': 1., 'sin': 0.},
            ...                                 'operators': {'+': 0., '-': 1.}
            ...                             }
            ...                         },
            ...     'operator_conditionals': {
            ...                             '+': {
            ...                                 'features': {'constants': .5, 'variables': .5},
            ...                                 'functions': {'sin': 1., 'cos': 0.},
            ...                                 'operators': {'+': 1., '-': 0.}
            ...                             },
            ...                             '-': {
            ...                                 'features': {'constants': .3, 'variables': .7},
            ...                                 'functions': {'cos': .5, 'sin': .5},
            ...                                 'operators': {'+': .9, '-': .1}
            ...                             }
            ...                         },
            ... }
            >>> equation_tree = EquationTree.from_prior(p, 3)
            >>> equation_tree.structure
            [0, 1, 2]
            >>> equation_tree.expr
            ['cos', 'cos', 'x_1']
            >>> equation_tree = EquationTree.from_prior(p, 3)
            >>> equation_tree.structure
            [0, 1, 2]
            >>> equation_tree.expr
            ['sin', 'cos', 'x_1']
            >>> equation_tree = EquationTree.from_prior(p, 3)
            >>> equation_tree.structure
            [0, 1, 1]
            >>> equation_tree = EquationTree.from_prior(p, 3)
            >>> equation_tree.structure
            [0, 1, 2, 3, 2, 3, 1]
            >>> equation_tree.expr
            ['+', '+', 'sin', 'x_1', 'sin', 'x_2', 'x_2']
            >>> equation_tree.sympy_expr
            x_2 + sin(x_1) + sin(x_2)

            # Without conditionals, the unconditioned priors are the fallback option
            >>> p = {
            ...     'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
            ...     'features': {'constants': .2, 'variables': .8},
            ...     'functions': {'sin': .5, 'cos': .5},
            ...     'operators': {'+': .5, '-': .5},
            ... }
            >>> equation_tree = EquationTree.from_prior(p, 3)
            >>> equation_tree.structure
            [0, 1, 2, 3, 2, 3, 1]
            >>> equation_tree.expr
            ['+', '-', 'cos', 'c_1', 'cos', 'c_2', 'c_3']
            >>> equation_tree.sympy_expr
            c_3 + cos(c_1) - cos(c_2)

            # Note: this would be discarded in a future step as unnecesarry complex
        """
        root = sample_tree_full(prior, max_variables_unique)
        return cls(root)

    @classmethod
    def from_priors(
            cls,
            feature_priors={},
            function_priors={},
            operator_priors={},
            structure_priors={},
    ):
        """
        Instantiate a tree from priors

        Attention
            - use standard notation here:   variables should be in form x_{number}
                                            constants should be in form c_{number}

        Args:
            max_depth: Maximum depth of the tree
            feature_priors: The priors for the features (variables + constants)
            function_priors: The priors for the functions
            operator_priors: The priors for the operators
            structure_priors: The priors for the tree structures

        Example:
            >>> np.random.seed(42)
            >>> max_depth = 12
            >>> feature_priors = {"x_1": 0.5, "c_1": 0.5}
            >>> function_priors = {"sin": 0.5, "cos": 0.5}
            >>> operator_priors = {"+": 0.5, "-": 0.5}
            >>> structure_priors = {'[0, 1, 2, 3, 4, 5, 5, 2, 3, 4, 4]': 1}
            >>> equation_tree = EquationTree.from_priors(
            ...     feature_priors, function_priors, operator_priors, structure_priors)
            >>> equation_tree.expr
            ['cos', '-', 'cos', 'sin', '+', 'x_1', 'c_1', 'cos', '-', 'x_1', 'c_1']
            >>> equation_tree.structure
            [0, 1, 2, 3, 4, 5, 5, 2, 3, 4, 4]
            >>> equation_tree.variables
            ['x_1', 'x_1']
            >>> equation_tree.n_variables
            2
            >>> equation_tree.n_variables_unique
            1
            >>> equation_tree.constants
            ['c_1', 'c_1']
            >>> equation_tree.n_constants
            2
            >>> equation_tree.n_constants_unique
            1
            >>> equation_tree.n_leafs
            4
            >>> equation_tree.operators
            ['-', '+', '-']
            >>> equation_tree.functions
            ['cos', 'cos', 'sin', 'cos']
        """
        root = sample_tree(
            feature_priors,
            function_priors,
            operator_priors,
            structure_priors,
        )
        return cls(root)

    @classmethod
    def from_sympy(
            cls,
            expression,
            function_test: Callable = lambda _: False,
            operator_test: Callable = lambda _: False,
            variable_test: Callable = lambda _: False,
            constant_test: Callable = lambda _: False,
    ):
        """
        Instantiate a tree from a sympy function

        Attention:
            - constant and variable names get standardized
            - unary minus get converted to binary minus

        Examples:
            >>> expr = sympify('x_a + B * y')
            >>> expr
            B*y + x_a
            >>> is_operator = lambda x : x in ['+', '*']
            >>> is_variable = lambda x : '_' in x or x in ['y']
            >>> is_constant = lambda x : x == 'B'
            >>> equation_tree = EquationTree.from_sympy(
            ...     expr,
            ...     operator_test=is_operator,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant
            ... )
            >>> equation_tree.expr
            ['+', '*', 'c_1', 'x_2', 'x_1']
            >>> equation_tree.sympy_expr
            c_1*x_2 + x_1

            # Numbers don't get standardized but are constants
            >>> expr = sympify('x_a + 2 * y')
            >>> expr
            x_a + 2*y
            >>> equation_tree = EquationTree.from_sympy(
            ...     expr,
            ...     operator_test=is_operator,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant
            ... )
            >>> equation_tree.expr
            ['+', 'x_1', '*', '2', 'x_2']
            >>> equation_tree.sympy_expr
            x_1 + 2*x_2
            >>> is_operator = lambda x : x in ['+', '*', '**']
            >>> is_variable = lambda x: '_' in x
            >>> is_constant = lambda x: x == 'B'
            >>> equation_tree = EquationTree.from_sympy(
            ...     sympify('B*x_1**2'),
            ...     operator_test=is_operator,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant
            ... )

        """
        standard = standardize_sympy(expression, variable_test, constant_test)
        standard = unary_minus_to_binary(standard, operator_test)
        prefix = infix_to_prefix(str(standard), function_test, operator_test)
        root = node_from_prefix(
            prefix,
            function_test,
            operator_test,
            lambda x: "x_" in x,
            lambda x: "c_" in x,
        )
        return cls(root)

    @property
    def constants_unique(self):
        return list(set(self.constants))

    @property
    def n_constants(self):
        return len(self.constants)

    @property
    def n_constants_unique(self):
        return len(set(self.constants))

    @property
    def non_numeric_constants(self):
        return [
            c for c in self.constants if not is_numeric(c) and not is_known_constant(c)
        ]

    @property
    def non_numeric_constants_unique(self):
        return list(set(self.non_numeric_constants))

    @property
    def n_non_numeric_constants_unique(self):
        return len(self.non_numeric_constants_unique)

    @property
    def n_non_numeric_constants(self):
        return len(self.non_numeric_constants)

    @property
    def variables_unique(self):
        return list(set(self.variables))

    @property
    def n_variables(self):
        return len(self.variables)

    @property
    def n_variables_unique(self):
        return len(set(self.variables))

    @property
    def functions_unique(self):
        return list(set(self.functions))

    @property
    def operators_unique(self):
        return list(set(self.operators))

    @property
    def n_leafs(self):
        return self.n_constants + self.n_variables

    # TODO: implement this
    @property
    def standard_structure(self):
        return self.structure

    @property
    def prefix(self):
        return self.expr

    @property
    def infix(self):
        return prefix_to_infix(
            self.prefix, lambda x: x in self.functions, lambda x: x in self.operators
        )

    @property
    def sympy_expr(self):
        sympy_expr = sympify(self.infix)
        if sympy_expr.free_symbols:
            symbol_names = [str(symbol) for symbol in sympy_expr.free_symbols]
            real_symbols = symbols(" ".join(symbol_names), real=True)
            if not isinstance(real_symbols, list) and not isinstance(
                    real_symbols, tuple
            ):
                real_symbols = [real_symbols]
            subs_dict = {old: new for old, new in zip(symbol_names, real_symbols)}
            sympy_expr = sympy_expr.subs(subs_dict)
        return sympy_expr

    @property
    def is_nan(self):
        return self.root is None

    @property
    def value_samples_as_df(self):
        if self.evaluation is None:
            warnings.warn(
                "Tree not yet evaluated. Use method get_evaluation to evaluate the tree"
            )
            return None
        data = {"observation": self.evaluation[:, 0]}
        for idx, key in enumerate(self.expr):
            if key in self.variables:
                data[key] = self.evaluation[:, idx]
            if key in self.non_numeric_constants:
                data[key] = self.evaluation[:, idx]
        return pd.DataFrame(data)

    @property
    def has_valid_value(self):
        if self.evaluation is None:
            warnings.warn(
                "Tree not yet evaluated. Use method get_evaluation to evaluate the tree"
            )
            return False
        ev = self.evaluation[0, :]
        return np.any(np.isfinite(ev) & ~np.isnan(ev))

    @property
    def info(self):
        """
        Get al information as dictionary
        """
        info = {}
        info["max_depth"] = len(self.structure)
        info["depth"] = max(self.structure)
        info["structures"] = self.structure
        info["features"] = {
            "constants": self.n_constants,
            "variables": self.n_variables,
        }
        functions = {}
        function_conditionals = {key: {} for key in self.functions_unique}
        for f in self.functions_unique:
            functions[f] = len([_f for _f in self.functions if _f == f])
            p_functions, p_operators, p_features = self._get_conditionals(f)
            function_conditionals[f]["functions"] = p_functions
            function_conditionals[f]["operators"] = p_operators
            function_conditionals[f]["features"] = p_features
        operators = {}
        operator_conditionals = {key: {} for key in self.operators_unique}
        for o in self.operators_unique:
            operators[o] = len([_o for _o in self.operators if _o == o])
            p_functions, p_operators, p_features = self._get_conditionals(o)
            operator_conditionals[o]["functions"] = p_functions
            operator_conditionals[o]["operators"] = p_operators
            operator_conditionals[o]["features"] = p_features
        info["functions"] = functions
        info["function_conditionals"] = function_conditionals
        info["operators"] = operators
        info["operator_conditionals"] = operator_conditionals
        return info

    def evaluate(self, variables: Union[dict, pd.DataFrame]):
        """
        Examples:
            >>> expr = sympify('x_a + 3 * y')
            >>> is_operator = lambda x : x in ['+', '*']
            >>> is_variable = lambda x : '_' in x or x in ['y']
            >>> equation_tree = EquationTree.from_sympy(
            ...     expr,
            ...     operator_test=is_operator,
            ...     variable_test=is_variable,
            ... )
            >>> equation_tree.sympy_expr
            x_1 + 3*x_2

            # We can use dicts:
            >>> equation_tree.evaluate({'x_1': np.array([2, 3]), 'x_2': np.array([1, 1])})
            0    5
            1    6
            dtype: int64

            # Or pandas dataframe:
            >>> dataFrame = pd.DataFrame({'x_1': np.array([2, 3]), 'x_2': np.array([1, 1])})
            >>> dataFrame
               x_1  x_2
            0    2    1
            1    3    1
            >>> equation_tree.evaluate(dataFrame)
            0    5
            1    6
            dtype: int64

            # Or pandas dataframe:
            >>> dataFrame = pd.DataFrame({'x_1': [2, 3], 'x_2': [1, 1]})
            >>> dataFrame
               x_1  x_2
            0    2    1
            1    3    1
            >>> equation_tree.evaluate(dataFrame)
            0    5
            1    6
            dtype: int64
        """

        df = pd.DataFrame(variables)
        symbol_list = list(self.sympy_expr.free_symbols)

        if set(list(df.columns)).issubset(set(symbol_list)):
            raise Exception(f"Variables in expression {self.sympy_expr} "
                            f"do not match the given ones: {df.columns}")

        symbol_names = [str(s) for s in symbol_list]

        # Step 3: Convert the expression to a function
        f = sympy.lambdify(symbol_list, self.sympy_expr, "numpy")
        return np.array(f(*[df[name] for name in symbol_names]))

    def save_samples(self,
                     path,
                     num_samples,
                     ranges: Optional[Dict] = None,
                     default_range: float = 10,
                     dv_name: str = 'y',
                     random_state: Optional[int] = None,
                     compression: str = 'gzip',
                     ):
        """
        Creates a file with samples of ivs and dvs
        Args:
            path: The path were to store the file
            num_samples: The number of samples
            ranges: A dictionary with the ranges for the variables in form of a dict
            default_range: Default range to fall back to if no range for a
                specific variable is given
            dv_name: The name to give to the observation
            random_state: The random seed to be used
            compression: Compression method
        """
        if not path.endswith('gz') and compression == 'gzip':
            warnings.warn(f'Compression is gzip but file {path} does not have the ending .gz')
        _ranges = {key: (-default_range, default_range) for key in self.variables_unique}
        if ranges is not None:
            for key in ranges.keys():
                if key in _ranges.keys():
                    _ranges[key] = ranges[key]

        rng = np.random.default_rng(random_state)

        def _get_conditions_once():
            raw_conditions = {}
            for key in _ranges.keys():
                raw_conditions[key] = rng.uniform(*_ranges[key], size=num_samples)
            return pd.DataFrame(raw_conditions)

        conditions_ = pd.DataFrame(columns=self.variables_unique + [dv_name])
        i = 0
        while i < 1_000_000 and len(conditions_.index) < num_samples:
            _sample = _get_conditions_once()
            evaluation = self.evaluate(_sample)
            _sample[dv_name] = evaluation
            bad_indices = np.where(np.isnan(evaluation) | np.isinf(evaluation))[0]
            _sample = _sample.drop(bad_indices)
            conditions_ = pd.concat([conditions_, _sample], ignore_index=True)
            i += 1
            if i >= 1_000_000:
                break
        conditions_ = conditions_.head(num_samples)
        conditions_.to_csv(path, compression=compression, index=False, sep="\t")

    def save_samples_srbench(self,
                             path,
                             num_samples,
                             ranges: Optional[Dict] = None,
                             default_range: float = 10,
                             random_state: Optional[int] = None,
                             ):
        self.save_samples(path, num_samples, ranges, default_range, 'target', random_state)

    def save_meta_srbench(self,
                          path,
                          name_dataset,
                          name_target="y"):
        info = {}
        info["dataset"] = name_dataset
        info["description"] = name_target + " = " + str(self.sympy_expr)
        info["source"] = "https://github.com/AutoResearch/equation-tree"
        info["publication"] = "Not yet implemented"
        info["task"] = "regression"
        info["keywords"] = ["abstract", "math", "symbolic regression"]
        info["target"] = {"type": "continuous", "description": "abstract dependent variable"}
        info["features"] = [
            {"name": v, "type": "continuous", "description": "abstract independent variable"}
            for v in self.variables_unique]

        with open(path, "w") as f:
            yaml.dump(info, f, sort_keys=False)

    def export_to_srbench(self,
                          folder: str,
                          num_samples: int = 1000,
                          name_target: str = "y",
                          ranges: Optional[Dict] = None,
                          default_range: float = 10,
                          random_state: Optional[int] = None,
                          ):
        """
        Creates a folder and adds data and metadata to the folder that can be used with sr bench:
        https://cavalab.org/srbench/
        """
        os.mkdir(folder)
        path_data = f"{folder}/data.tsv.gz"
        path_meta = f"{folder}/metadata.yaml"
        self.save_samples_srbench(path_data, num_samples, ranges, default_range, random_state)
        self.save_meta_srbench(path_meta, "data", name_target)

    def check_validity(
            self,
            zero_representations=["0"],
            log_representations=["log", "Log"],
            division_representations=["/", ":"],
            verbose=False,
    ):
        """
        Check if the tree is valid:
            - Check if log(0) or x / 0 exists
            - Check if function(constant) or operator(constant_1, constant_2) exists
                    -> unnecessary complexity
            - Check if each function has exactly one child
            - Check if each operator has exactly two children

        Args:
            zero_representations: A list of attributes that represent zero
            log_representations: A list of attributes that represent log
            division_representations: A list of attributes that represent division
            verbose: If set true, print out the reason for the invalid tree

        Example:
            >>> is_variable = lambda x : x == 'x'
            >>> is_constant = lambda x : x == 'c' or x == '0'
            >>> is_operator = lambda x : x == '/'
            >>> equation_tree = EquationTree.from_prefix(
            ...     ['/', 'x', '0'],
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     operator_test=is_operator,
            ... )
            >>> equation_tree.check_validity()
            False
            >>> equation_tree.check_validity(verbose=True)
            division by 0 is not allowed.
            False
            >>> equation_tree = EquationTree.from_prefix(
            ...     ['/', 'x', 'c'],
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     operator_test=is_operator,
            ... )
            >>> equation_tree.check_validity()
            True

            >>> equation_tree = EquationTree.from_prefix(
            ...     ['/', '0', 'c'],
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     operator_test=is_operator,
            ... )
            >>> equation_tree.check_validity(verbose=True)
            0 and c are constants applied to the operator /
            False
        """
        return self.root.check_validity(
            zero_representations, log_representations, division_representations, verbose
        )

    def check_possible(
            self,
            feature_priors: Dict,
            function_priors: Dict,
            operator_priors: Dict,
            structure_priors: Dict,
    ):
        """
        Check weather a tree is a possible tree given the priors
        Attention:
            If no prior is given, interpreted as all possibilities are allowed
        """
        if feature_priors != {}:
            for c in self.constants:
                if c not in feature_priors.keys() or feature_priors[c] <= 0:
                    return False
            for v in self.variables:
                if v not in feature_priors.keys() or feature_priors[v] <= 0:
                    return False
        if function_priors != {}:
            for fun in self.functions:
                if fun not in function_priors.keys() or function_priors[fun] <= 0:
                    return False
        if operator_priors != {}:
            for op in self.operators:
                if op not in operator_priors.keys() or operator_priors[op] <= 0:
                    return False
        if structure_priors != {}:
            if (
                    str(self.structure) not in structure_priors.keys()
                    or structure_priors[str(self.structure)] <= 0
            ):
                return False
        return True

    def check_possible_from_prior(self, prior: Dict):
        structure_priors = prior["structures"] if "structures" in prior.keys() else {}
        feature_priors = prior["features"] if "features" in prior.keys() else {}
        function_priors = prior["functions"] if "functions" in prior.keys() else {}
        operator_priors = prior["operators"] if "operators" in prior.keys() else {}
        function_conditionals = (
            prior["function_conditionals"]
            if "function_conditionals" in prior.keys()
            else {}
        )
        operator_conditionals = (
            prior["operator_conditionals"]
            if "operator_conditionals" in prior.keys()
            else {}
        )
        if structure_priors:
            if (
                    str(self.structure) not in structure_priors.keys()
                    or structure_priors[str(self.structure)] <= 0
            ):
                return False
        if feature_priors:
            if (
                    self.n_variables > 0
                    and (
                            "variables" not in feature_priors.keys()
                            or feature_priors["variables"] <= 0
                    )
            ) or (
                    self.n_constants > 0
                    and (
                            "constants" not in feature_priors.keys()
                            or feature_priors["constants"] <= 0
                    )
            ):
                return False
        if function_priors:
            for fun in self.functions_unique:
                if fun not in function_priors.keys() or function_priors[fun] <= 0:
                    return False

        if operator_priors:
            for op in self.operators_unique:
                if op not in operator_priors.keys() or operator_priors[op] <= 0:
                    return False

        test_lst = []

        def test_node(child):
            nonlocal test_lst
            if child is None or child.parent is None:
                test_lst.append(True)
                return
            node = child.parent
            _prior = {}
            if (
                    node.kind == NodeKind.FUNCTION
                    and node.attribute in function_conditionals.keys()
            ):
                _prior = function_conditionals[node.attribute]
            elif (
                    node.kind == NodeKind.OPERATOR
                    and node.attribute in operator_conditionals.keys()
            ):
                _prior = operator_conditionals[node.attribute]
            if child.kind == NodeKind.FUNCTION:
                _p = _prior[node.attribute]["functions"]
                if (
                        _p != {}
                        and node.attribute not in _p.keys()
                        or _p[node.attribute] <= 0
                ):
                    test_lst.append(False)
            elif child.kind == NodeKind.OPERATOR:
                _p = _prior[node.attribute]["operators"]
                if (
                        _p != {}
                        and node.attribute not in _p.keys()
                        or _p[node.attribute] <= 0
                ):
                    test_lst.append(False)
            else:
                _p = _prior["features"] if "features" in _prior else {}
                if child.kind == NodeKind.VARIABLE:
                    if (
                            _p != {}
                            and "variables" not in _p.keys()
                            or _p["variables"] <= 0
                    ):
                        test_lst.append(False)
                if child.kind == NodeKind.CONSTANT:
                    if (
                            _p != {}
                            and "constants" not in _p.keys()
                            or _p["constants"] <= 0
                    ):
                        test_lst.append(False)
            for c in child.children:
                test_node(c)

        conditionals = not any(item is False for item in test_lst)
        if not conditionals:
            return False
        return True

    def standardize(self):
        """
        Standardize variable and constant names

        Example:
            >>> is_variable = lambda x : x in ['x', 'y', 'z']
            >>> is_constant = lambda x : x in ['0', '1', '2']
            >>> is_function = lambda x : x in ['sin', 'cos']
            >>> is_operator = lambda x: x in ['+', '-', '*', '/']
            >>> prefix = ['+', '-', 'x', '1', '*', 'sin', 'y', 'cos', 'z']

            # then we create the node root
            >>> equation_tree = EquationTree.from_prefix(
            ...     prefix_notation=prefix,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     function_test=is_function,
            ...     operator_test=is_operator
            ...     )

            >>> equation_tree.sympy_expr
            x + sin(y)*cos(z) - 1

            >>> equation_tree.standardize()
            >>> equation_tree.sympy_expr
            x_1 + sin(x_2)*cos(x_3) - 1

        """
        variable_count = 0
        constant_count = 0
        variables = {}
        constants = {}

        def rec_stand(node):
            if node is None:
                return
            nonlocal variable_count, constant_count
            nonlocal variables, constants
            if node.kind == NodeKind.VARIABLE:
                if node.attribute not in variables.keys():
                    variable_count += 1
                    variables[node.attribute] = f"x_{variable_count}"
                node.attribute = variables[node.attribute]
            if node.kind == NodeKind.CONSTANT and not is_numeric(node.attribute):
                if node.attribute not in constants.keys():
                    constant_count += 1
                    constants[node.attribute] = f"c_{constant_count}"
                node.attribute = constants[node.attribute]
            else:
                rec_stand(node.left)
                rec_stand(node.right)
            return node

        self.root = rec_stand(self.root)
        self._build()

    def simplify(
            self,
            function_test: Union[Callable, None] = None,
            operator_test: Union[Callable, None] = None,
            is_binary_minus_only: bool = True,
            is_power_caret: bool = True,
            verbose: bool = False,
    ):
        """
        Simplify equation if the simplified equation has a shorter prefix
        Args:
            function_test: A function that tests weather an attribute is a function
                Attention: simplifying may lead to new functions that were not in the equation
                    before. If so, add this to the test here.
            operator_test: A function that tests weather an attribute is an operator
                Attention: simplifying may lead to new operators that were not in the equation
                    before. If so, add this to the test here.
            is_binary_minus_only: Convert all unary minus to binary after simplification
            is_power_caret: Represent power as a caret after simplification
            verbose: Show messages if simplification results in errors

        Examples:
            >>> is_variable = lambda x: 'x_' in x
            >>> is_constant = lambda x: 'c_' in x or is_numeric(x) or is_known_constant(x)
            >>> is_operator = lambda x: len(x) == 1 and not is_numeric(x)
            >>> is_function = lambda x: not (is_variable(x) or is_constant(x) or is_operator(x))
            >>> prefix_notation = ['+', 'x_1', 'x_1' ]
            >>> equation_tree = EquationTree.from_prefix(
            ...     prefix_notation=prefix_notation,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     operator_test=is_operator,
            ...     function_test=is_function)
            >>> equation_tree.expr
            ['+', 'x_1', 'x_1']

            # simplifying the expression without function test will result in an error since the
            # new function has an unknown operator:
            >>> equation_tree.simplify()
            Traceback (most recent call last):
            ...
            Exception: * has no defined type in any space

            # we can provide the same function that we used to generate the equation though since
            # it takes care of multiplication:
            >>> equation_tree.simplify(operator_test=is_operator)
            >>> equation_tree.expr
            ['*', '2', 'x_1']

            >>> prefix_notation = ['sqrt', '*', 'x_1', 'x_1']
            >>> equation_tree = EquationTree.from_prefix(
            ...     prefix_notation=prefix_notation,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     operator_test=is_operator,
            ...     function_test=is_function)
            >>> equation_tree.expr
            ['sqrt', '*', 'x_1', 'x_1']

            # it is good practice to define tests at the begining of a script and use them
            # throughout the project
            >>> equation_tree.simplify(
            ...     operator_test=is_operator,
            ...     function_test=is_function
            ... )
            >>> equation_tree.expr
            ['abs', 'x_1']

            >>> prefix_notation = ['*', '-', 'c_1', 'x_1', '-', 'x_1', 'c_1']
            >>> equation_tree = EquationTree.from_prefix(
            ...     prefix_notation=prefix_notation,
            ...     variable_test=is_variable,
            ...     constant_test=is_constant,
            ...     operator_test=is_operator,
            ...     function_test=is_function)
            >>> equation_tree.expr
            ['*', '-', 'c_1', 'x_1', '-', 'x_1', 'c_1']

            >>> equation_tree.sympy_expr
            (-c_1 + x_1)*(c_1 - x_1)

            # it is good practice to define tests at the begining of a script and use them
            # throughout the project
            >>> equation_tree.simplify(
            ...     operator_test=is_operator,
            ...     function_test=is_function
            ... )
            >>> equation_tree.expr
            ['^', '-', '0', '-', 'c_1', 'x_1', '2']
        """

        if function_test is None:

            def function_test(x):
                return x in self.functions

        else:
            tmp_f = function_test

            def function_test(x):
                return tmp_f(x) or x in self.functions

        if operator_test is None:

            def operator_test(x):
                return x in self.operators

        else:
            tmp_o = operator_test

            def operator_test(x):
                return tmp_o(x) or x in self.operators

        simplified_equation = simplify(self.sympy_expr)
        if not check_functions(simplified_equation, function_test):
            warnings.warn(
                f"{simplified_equation} has functions that are not in function_test type"
            )
            self.root = None
            self._build()
            return
        if (
                "I" in str(simplified_equation)
                or "accumbounds" in str(simplified_equation).lower()
        ):
            if verbose:
                print(f"Simplify {str(self.sympy_expr)} results in complex values")
            self.root = None
            self._build()
            return
        if is_power_caret:
            simplified_equation = str(simplified_equation).replace("**", "^")
        simplified_equation = simplified_equation.replace("re", "")
        if is_binary_minus_only:
            simplified_equation = unary_minus_to_binary(
                simplified_equation, operator_test
            )
        simplified_equation = simplified_equation.replace(" ", "")

        prefix = infix_to_prefix(simplified_equation, function_test, operator_test)
        if verbose:
            print("prefix", simplified_equation)
            print("prefix tree", prefix)
        if "re" in prefix:
            prefix.remove("re")
        if len(prefix) > len(self.expr):
            prefix = self.expr
        if "zoo" in prefix or "oo" in prefix:
            if verbose:
                print(f"Simplify {str(self.sympy_expr)} results in None")
            self.root = None
            self._build()
            return
        self.root = node_from_prefix(
            prefix,
            function_test,
            operator_test,
            lambda x: x in self.variables,
            lambda x: x in self.constants or is_numeric(x) or is_known_constant(x),
        )
        self._build()

    def get_evaluation(
            self, min_val: int = -1, max_val: int = 1, num_samples: int = 100
    ):
        """
        Evaluate the nodes with random samples for variables and constants.
        """

        crossings = self._create_crossing(min_val, max_val, num_samples)
        evaluation = np.zeros((len(crossings), len(self.expr)))

        for i, crossing in enumerate(crossings):
            eqn_input = dict()
            k = 0
            for c in self.constants_unique:
                if is_numeric(c):
                    eqn_input[c] = float(c)
                elif c == "e":
                    eqn_input[c] = 2.71828182846
                elif c == "pi":
                    eqn_input[c] = 3.14159265359
                else:
                    eqn_input[c] = crossing[self.n_variables_unique + k]
                    k += 1
            for idx, x in enumerate(self.variables_unique):
                eqn_input[x] = crossing[idx]
            evaluation[i, :] = self._evaluate(eqn_input)

        self.evaluation = evaluation
        return evaluation

    def _evaluate(self, features: Dict):
        values: List[float] = list()

        if self.root is not None:
            self._evaluate_node(features, self.root)
            values = self._get_full_evaluation(self.root)

        return values

    def _evaluate_node(self, features: Dict, node: TreeNode):
        if node.kind == NodeKind.FUNCTION:
            if node.left is None:
                raise Exception("Invalid tree: %s" % self.expr)
            value = FUNCTIONS[node.attribute](self._evaluate_node(features, node.left))

        elif node.kind == NodeKind.OPERATOR:
            if node.left is None or node.right is None:
                raise Exception("Invalid tree: %s" % self.expr)
            value = OPERATORS[node.attribute](
                self._evaluate_node(features, node.left),
                self._evaluate_node(features, node.right),
            )

        elif node.kind == NodeKind.CONSTANT or node.kind == NodeKind.VARIABLE:
            value = features[node.attribute]
        else:
            raise Exception("Invalid attribute %s" % node.attribute)
        node.evaluation = value
        return value

    def _get_full_evaluation(self, node: TreeNode):
        values = list()
        values.append(node.evaluation)

        if node.kind == NodeKind.FUNCTION:
            if node.left is None:
                raise Exception("Invalid tree: %s" % self.expr)
            eval_add = self._get_full_evaluation(node.left)
            for eval_element in eval_add:
                values.append(eval_element)

        if node.kind == NodeKind.OPERATOR:
            if node.left is None or node.right is None:
                raise Exception("Invalid tree: %s" % self.expr)
            eval_add = self._get_full_evaluation(node.left)
            for eval_element in eval_add:
                values.append(eval_element)
            eval_add = self._get_full_evaluation(node.right)
            for eval_element in eval_add:
                values.append(eval_element)

        return values

    def _create_crossing(
            self, min_val: float = -1, max_val: float = 1, num_samples: int = 100
    ):
        crossings = []

        total_unique = self.n_variables_unique + self.n_non_numeric_constants_unique

        for _ in range(num_samples):
            sample = []
            for _ in range(total_unique):
                value = np.random.uniform(min_val, max_val)
                sample.append(value)
            crossings.append(sample)

        return np.array(crossings)

    def _build(self):
        self.structure: List[int] = []

        # make function to get this here
        self.expr: List[str] = list()

        self.variables: List[str] = list()
        self.functions: List[str] = list()
        self.operators: List[str] = list()
        self.constants: List[str] = list()

        self.evaluation = None

        self._collect_structure(self.structure, 0, self.root)

        self._collect_attributes(
            lambda node: node.kind == NodeKind.VARIABLE, self.variables, self.root
        )
        self._collect_attributes(
            lambda node: node.kind == NodeKind.FUNCTION, self.functions, self.root
        )
        self._collect_attributes(
            lambda node: node.kind == NodeKind.CONSTANT, self.constants, self.root
        )
        self._collect_attributes(
            lambda node: node.kind == NodeKind.OPERATOR, self.operators, self.root
        )
        self._collect_expr(self.expr, self.root)

    def _collect_structure(self, structure=[], level=0, node=None):
        if node is None:
            return
        structure.append(level)
        self._collect_structure(structure, level + 1, node.left)
        self._collect_structure(structure, level + 1, node.right)
        return

    def _collect_expr(self, expression=[], node=None):
        if node is None:
            return
        expression.append(node.attribute)
        self._collect_expr(expression, node.left)
        self._collect_expr(expression, node.right)

    def _collect_attributes(
            self, attribute_identifier: Callable = lambda _: True, attributes=[], node=None
    ):
        if node is None:
            return list()
        if attribute_identifier(node):
            attributes.append(node.attribute)
        if node.left is not None:
            self._collect_attributes(attribute_identifier, attributes, node.left)
        if node.right is not None:
            self._collect_attributes(attribute_identifier, attributes, node.right)
        return attributes

    def _get_conditionals(self, attribute):
        functions = {}
        operators = {}
        features = {"constants": 0, "variables": 0}

        def get_child(node):
            nonlocal functions, operators, features
            if node is None:
                return
            if node.attribute == attribute:
                if node.kind == NodeKind.FUNCTION or NodeKind.OPERATOR:
                    if node.left.kind == NodeKind.FUNCTION:
                        if node.left.attribute in functions.keys():
                            functions[node.left.attribute] += 1
                        else:
                            functions[node.left.attribute] = 1
                    if node.left.kind == NodeKind.OPERATOR:
                        if node.left.attribute in operators.keys():
                            operators[node.left.attribute] += 1
                        else:
                            operators[node.left.attribute] = 1
                    if node.left.kind == NodeKind.CONSTANT:
                        features["constants"] += 1
                    if node.left.kind == NodeKind.VARIABLE:
                        features["variables"] += 1
                if node.kind == NodeKind.OPERATOR:
                    if node.right.kind == NodeKind.FUNCTION:
                        if node.right.attribute in functions.keys():
                            functions[node.right.attribute] += 1
                        else:
                            functions[node.right.attribute] = 1
                    if node.right.kind == NodeKind.OPERATOR:
                        if node.right.attribute in operators.keys():
                            operators[node.right.attribute] += 1
                        else:
                            operators[node.right.attribute] = 1
                    if node.right.kind == NodeKind.CONSTANT:
                        features["constants"] += 1
                    if node.right.kind == NodeKind.VARIABLE:
                        features["variables"] += 1
            get_child(node.left)
            get_child(node.right)

        get_child(self.root)

        return functions, operators, features


def instantiate_constants(tree, fct: Callable):
    """
    Examples:
        >>> expr = sympify('e ** (- e **( x + y))')
        >>> is_operator = lambda x : x in ['+', '*', '**', '-']
        >>> is_variable = lambda x : x in ['x', 'y']
        >>> is_constant = lambda x : x in ['e']
        >>> equation_tree = EquationTree.from_sympy(
        ...     expr,
        ...     operator_test=is_operator,
        ...     variable_test=is_variable,
        ... )
        >>> equation_tree.sympy_expr
        e**((-e)**(x_1 + x_2))
        >>> instantiated = instantiate_constants(equation_tree, lambda: 2)
        >>> instantiated.sympy_expr
        e**((-e)**(x_1 + x_2))
        >>> expr_2 = sympify('c_1 ** (- e **( x + y))')
        >>> equation_tree_2 = EquationTree.from_sympy(
        ...     expr_2,
        ...     operator_test=is_operator,
        ...     variable_test=is_variable,
        ... )
        >>> instantiated = instantiate_constants(equation_tree_2, lambda: 2)
        >>> instantiated.sympy_expr
        2**((-e)**(x_1 + x_2))



    """
    root = copy.deepcopy(tree.root)

    def _rec_apply(node):
        if (node.kind == NodeKind.CONSTANT
                and not is_known_constant(node.attribute)
                and not is_numeric(node.attribute)):
            node.attribute = str(fct())
        for c in node.children:
            _rec_apply(c)

    _rec_apply(root)
    return EquationTree(root)
