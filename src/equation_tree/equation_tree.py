from typing import Callable, Dict, List, Union

from src.tree_node import TreeNode, node_from_prefix, NodeKind, sample_tree
from util.conversions import prefix_to_infix, infix_to_prefix, unary_minus_to_binary, \
    standardize_sympy

import numpy as np
from sympy import sympify, simplify, symbols

UnaryOperator = Callable[[Union[int, float]], Union[int, float]]
BinaryOperator = Callable[[Union[int, float], Union[int, float]], Union[int, float]]

operators: Dict[str, BinaryOperator] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "^": lambda a, b: a ** b,
}

functions: Dict[str, UnaryOperator] = {
    "sin": lambda a: np.sin(a),
    "cos": lambda a: np.cos(a),
    "tan": lambda a: np.tan(a),
    "exp": lambda a: np.exp(a),
    "log": lambda a: np.log(a),
    "sqrt": lambda a: np.sqrt(a),
    "abs": lambda a: np.abs(a),
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
            >>> feature_priors = {"x_1": 0.5, "c_1": 0.5}
            >>> function_priors = {"sin": 0.5, "cos": 0.5}
            >>> operator_priors = {"+": 0.5, "-": 0.5}
            >>> node_root = sample_tree(max_depth, feature_priors, function_priors, operator_priors)
            >>> equation_tree = EquationTree(node_root)
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

        self.root: TreeNode = node

        # make function to get tree structure here
        self.structure: List[int] = []

        # make function to get this here
        self.expr: List[str] = list()

        self.variables: List[str] = list()
        self.functions: List[str] = list()
        self.operators: List[str] = list()
        self.constants: List[str] = list()

        self._build()

    @classmethod
    def from_prefix(cls,
                    prefix_notation: List[str],
                    function_test: Callable = lambda _: False,
                    operator_test: Callable = lambda _: False,
                    variable_test: Callable = lambda _: False,
                    constant_test: Callable = lambda _: False):
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
        root = node_from_prefix(prefix_notation, function_test, operator_test, variable_test,
                                constant_test)
        return cls(root)

    @classmethod
    def from_priors(cls,
                    max_depth,
                    feature_priors={},
                    function_priors={},
                    operator_priors={},
                    structure_priors={},
                    ):
        """Instantiate a tree from priors

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
            >>> equation_tree = EquationTree.from_priors(max_depth,
            ...     feature_priors, function_priors, operator_priors)
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
        root = sample_tree(max_depth, feature_priors, function_priors, operator_priors,
                           structure_priors)
        return cls(root)

    @classmethod
    def from_sympy(cls,
                   expression,
                   function_test: Callable = lambda _: False,
                   operator_test: Callable = lambda _: False,
                   variable_test: Callable = lambda _: False,
                   constant_test: Callable = lambda _: False):
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

        """
        standard = standardize_sympy(expression, variable_test, constant_test)
        standard = unary_minus_to_binary(standard, operator_test)
        prefix = infix_to_prefix(str(standard), function_test, operator_test)
        root = node_from_prefix(prefix, function_test, operator_test, variable_test,
                           constant_test)
        return cls(root)

    @property
    def n_constants(self):
        return len(self.constants)

    @property
    def n_constants_unique(self):
        return len(set(self.constants))

    @property
    def n_variables(self):
        return len(self.variables)

    @property
    def n_variables_unique(self):
        return len(set(self.variables))

    @property
    def n_leafs(self):
        return self.n_constants + self.n_variables

    @property
    def prefix(self):
        return self.expr

    @property
    def infix(self):
        return prefix_to_infix(
            self.prefix, lambda x: x in self.functions, lambda x: x in self.operators)

    @property
    def sympy_expr(self):
        sympy_expr = sympify(self.infix)
        if sympy_expr.free_symbols:
            symbol_names = [str(symbol) for symbol in sympy_expr.free_symbols]
            real_symbols = symbols(" ".join(symbol_names), real=True)
            if not isinstance(real_symbols, list) and not isinstance(real_symbols, tuple):
                real_symbols = [real_symbols]
            subs_dict = {old: new for old, new in zip(symbol_names, real_symbols)}
            sympy_expr = sympy_expr.subs(subs_dict)
        return sympy_expr

    def check_validity(self, zero_representations=['0'],
                       log_representations=['log', 'Log'],
                       division_representations=['/', ':'],
                       verbose=False):
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
            zero_representations,
            log_representations,
            division_representations,
            verbose
        )

    def standardize(self):
        pass

    # def simplify_tree(self,
    #                   function_test: Callable,
    #                   operator_test: Callable,
    #                   is_unary_minus_only: bool = True,
    #                   is_power_caret: bool = True,
    #                   verbose: bool = True):
    #     simplified_equation = simplify(self.sympy_expr)
    #     if I in simplified_equation.free_symbols:
    #         if verbose:
    #             print(f'Simplify {str(self.sympy_expr)} results in complex values')
    #         self.__init__(None)
    #         return
    #     if is_unary_minus_only:
    #         simplified_equation = unary_minus_to_binary(
    #             str(simplified_equation), lambda x: x in self.operators
    #         )
    #
    #     simplified_equation = simplified_equation.replace(" ", "")
    #     if is_power_caret:
    #         simplified_equation = simplified_equation.replace("**", "^")
    #
    #     prefix = infix_to_prefix(simplified_equation, function_test, operator_test)
    #     if verbose:
    #         print("prefix", simplified_equation)
    #         print("prefix tree", prefix)
    #     if len(prefix) > len(expr):
    #         prefix = expr
    #     if "re" in prefix:
    #         prefix.remove("re")
    #     if "zoo" in prefix or "oo" in prefix:
    #         return None
    #     tree = EquationTree([], feature_space, function_space, operator_space)
    #     tree.instantiate_from_prefix_notation(prefix)
    #     return tree


    def _build(self):
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


    def evaluate(self, features: Dict):
        eval: List[float] = list()

        if self.root is not None:
            self.evaluate_node(features, self.root)
            eval = self.get_full_evaluation(self.root)

        return eval


    def get_full_evaluation(self, node: TreeNode):
        eval = list()
        eval.append(node.evaluation)

        if node.kind == NodeKind.FUNCTION:
            if node.left is None:
                raise Exception("Invalid tree: Left child of function node is None")
            eval_add = self.get_full_evaluation(node.left)
            for eval_element in eval_add:
                eval.append(eval_element)

        if node.kind == NodeKind.OPERATOR:
            if node.left is None or node.right is None:
                raise Exception("Invalid tree: operator node does not have 2 children")
            eval_add = self.get_full_evaluation(node.left)
            for eval_element in eval_add:
                eval.append(eval_element)
            eval_add = self.get_full_evaluation(node.right)
            for eval_element in eval_add:
                eval.append(eval_element)
        return eval


    def evaluate_node(self, features: Dict, node: NodeKind):
        if node is None:
            if self.root is not None:
                value = self.evaluate_node(features, self.root)
            else:
                value = 0

        if node.type == NodeKind.FUNCTION:
            if node.left is None:
                raise Exception("Invalid tree: %s" % self.expr)
            value = functions[node.attribute](self.evaluate_node(features, node.left))

        elif node.type == NodeKind.OPERATOR:
            if node.left is None or node.right is None:
                raise Exception("Invalid tree: %s" % self.expr)
            value = operators[node.attribute](
                self.evaluate_node(features, node.left),
                self.evaluate_node(features, node.right),
            )

        elif node.attribute in features:
            value = features[node.attribute]

        else:
            raise Exception("Invalid attribute %s" % node.attribute)

        node.evaluation = value
        return value

