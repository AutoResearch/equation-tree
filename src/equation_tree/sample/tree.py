from typing import List, Dict

from equation_tree.sample.tree_structure import sample_tree_structure, _count_children, NodeType, \
    TreeNode, _get_children

import numpy as np


def sample_attribute(attribute_list: List[str], priors: Dict, parent_attribute=""):
    probabilities = np.ones(len(attribute_list))
    for idx, attribute in enumerate(attribute_list):
        if parent_attribute != "":
            key = parent_attribute + "_" + attribute
        else:
            key = attribute
        if key in priors:
            probabilities[idx] = priors[key]
    probabilities = probabilities / np.sum(probabilities)
    sample_index = np.random.choice(len(attribute_list), p=probabilities)
    return attribute_list[sample_index]


def sample_attribute_from_tree(tree_structure, index,
                               feature_space, function_space, operator_space,
                               feature_priors, function_priors, operator_priors,
                               parent_attribute=""):
    num_children = _count_children(tree_structure, index)
    if num_children == 0:
        return sample_attribute(
            feature_space, feature_priors, parent_attribute
        )
    elif num_children == 1:
        return sample_attribute(
            function_space, function_priors, parent_attribute
        )
    elif num_children == 2:
        return sample_attribute(
            operator_space, operator_priors, parent_attribute
        )
    else:
        raise Exception("Invalid number of children: %s" % num_children)


def sample_equation_tree_from_structure(tree_structure, index,
                                        feature_space, function_space, operator_space,
                                        feature_priors, function_priors, operator_priors,
                                        parent_attribute=""):

    attribute = sample_attribute_from_tree(
        tree_structure, index,
        feature_space, function_space, operator_space,
        feature_priors, function_priors, operator_priors,
        parent_attribute)

    kind = NodeType.NONE

    if attribute in function_space:
        kind = NodeType.FUNCTION
    elif attribute in operator_space:
        kind = NodeType.OPERATION
    elif attribute in feature_space:
        if "x_" in attribute:
            kind = NodeType.INPUT
        elif "c_" in attribute:
            kind = NodeType.CONSTANT
        elif "0" in attribute:
            kind = NodeType.CONSTANT

    print(kind)

    node = TreeNode(val=tree_structure[index], attribute=attribute, type=kind)

    children = _get_children(tree_structure, index)

    if len(children) >= 1:
        node.left = sample_equation_tree_from_structure(
            tree_structure, children[0],
            feature_space, function_space, operator_space,
            feature_priors, function_priors, operator_priors,
            parent_attribute=attribute
        )

    if len(children) == 2:
        node.right = sample_equation_tree_from_structure(
            tree_structure, children[1],
            feature_space, function_space, operator_space,
            feature_priors, function_priors, operator_priors,
            parent_attribute=attribute
        )
    return node


def sample_tree(max_depth,
                feature_space, function_space, operator_space,
                feature_prior={}, function_prior={}, operator_prior={}, structure_prior={}):
    tree_structure = sample_tree_structure(max_depth=max_depth, prior=structure_prior)
    tree = sample_equation_tree_from_structure(
        tree_structure, 0,
        feature_space, function_space, operator_space,
        feature_prior, function_prior, operator_prior)
    return tree


