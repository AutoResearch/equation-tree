# Equation Sampling

![Equation Tree](../img/equation-sampler.gif)

## Sampling Features

The sampling of equations allows for dynamic benchmarking of symbolic regression algorithms. It also allows generating a large training dataset for deep learning symbolic regression.

Various features of the underlying equation distribution can be customized. For example, to mimic the distribution in specific scientific fields. This customization is implemented twofold:
- We can adjust equation **complexity**, dimension of the domain, and other attributes.
- We can use **prior frequency information** about the occurrence of specific operators (like +, -, and *) and functions (like sine, cosine, and logarithm).

## Method

In our sampling method, we distinct equation structure and equation content and sample both separately:
- First, in the *(1) Structure Sampling* step, we sample the structure of the underlying tree. Here, complexity is adjusted, and we can use prior information about structures.
- Second, in the *(2) Attribute Sampling* step, we sample the content of each tree node individually. Here, we can use prior information about the occurrence probabilities of specific operators and frequencies. This information can be conditioned on the parent nodes. For example, we can use prior information about the likelihood of + appearing in a sine function.

## How To Use The Sampler

To use our sampler, import the functionality and call the sample function: 
```python
from equation_tree import sample

equations = sample()
```
This will return a list of sampled equations. You can customize the number of equations and the dimension of the input via the keyword arguments `n` and `max_num_variables`. For example to sample 100 equations with a maximum of 3 input variables, write:
```python
equations = sample(n=100, max_num_variables=3)
```
The most versatile way to further customize the sampling is the use of a prior. You can pass this to the sampler as a dictionary with entries for a structures prior, features, functions and operators. Here, we give an example:
```python
prior = {
    'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3]': .4},
    'features': {'constants': .2, 'variables': .8},
    'functions': {'sin': .5, 'cos': .5},
    'operators': {'+': 1., '-': .0},
}

# To use the prior use the keyword argument `prior`
equations = sample(prior=prior)
```
You can also include conditionals. These influence the likelihood of a specific attribute being sampled given its parent node. For example, how likely does a - occur in a sine function:
```python
prior = {
    'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3]': .4},
    'features': {'constants': .2, 'variables': .8},
    'functions': {'sin': .5, 'cos': .5},
    'operators': {'+': 1., '-': .0},
        'function_conditionals': {
            'sin': {
                'features': {'constants': 0., 'variables': 1.},
                'functions': {'sin': 0., 'cos': 1.},
                'operators': {'+': 0., '-': 1.}
            },
            'cos': {
                'features': {'constants': 0., 'variables': 1.},
                'functions': {'cos': 1., 'sin': 0.},
                'operators': {'+': 0., '-': 1.}
            }
    },
        'operator_conditionals': {
            '+': {
                'features': {'constants': .5, 'variables': .5},
                'functions': {'sin': 1., 'cos': 0.},
                'operators': {'+': 1., '-': 0.}
            },
            '-': {
                'features': {'constants': .3, 'variables': .7},
                'functions': {'cos': .5, 'sin': .5},
                'operators': {'+': .9, '-': .1}
            }
    },
}
```

### Possible Attributes
Here, we present which attributes are supported natively. 

*You can use custom attributes for operators and functions, but other functionality like distance metrics or the evaluation of equations might not work with custom attributes.*

#### Structures
Here, we use the structure notion highlighted in the [format](equation-formats.md#tree-structure). 

The Equation Tree package provides convenience functions to obtain uniform structure priors from the tree depth or from the maximum number of nodes. To call them, you can use the keyword argument in the sample function: 
```python
# Sample equations with only a specified tree depth
equations = sample(depth=...)

# Sample equations up to a specified depth
equatons = sample(max_depth=...)
```


#### Features
`constants`: the likelihood of a leaf being a constant. In the Equation Tree package, constants are represented as c followed by an index (`c_{}`). The sampler doesn't sample the same constant twice. 

`variables`: the likelihood of a leaf being a variable. Variables are represented as a x followed by an index (`x_{}`). Variables are sampled with replacement. 

*Attention*: A function will never have a constant as it's child, since a constant in a function can be simplified to a single constant

#### Functions
Functions are mathematical operations with only one input value. Our package supports the following natively. Please ues the exact notion.
- sin
- cos
- tan
- exp
- log
- sqrt
- abs

The following operators can be added, but are not in the default priors:
- acos
- arg
- asin
- sinh
- cosh
- tanh
- cot

*Additionally, you can use `squared` and `cubed` as keys, but this might not be fully supported in all functions of the equation sampler. For example, converting to sympy expressions might lead to unexpected results.*

#### Operators
Operators are mathematical operations with two input values. Our package supports the following natively.
- \+
- \-
- \*
- \/
- \**
- max
- min

#### Conditionals
- In the function conditionals each function can has it's own prior consisting of a feature, function, and operator prior.
- In the operator conditionals each operator can has it's own prior consisting of a feature, function, and operator prior.

#### Convenience
The Equation Tree package has a convenience function that allows to transform a space into a uniform prior:
```python
from equation_tree.prior import prior_from_space

# For example if you only want to include primitive operators
operator_prior = prior_from_space(["+", "-", "*", "/"])
```

