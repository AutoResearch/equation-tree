from equation_tree.sample import burn
from equation_tree.defaults import DEFAULT_PRIOR

prior_0 = {
    'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
    'features': {'constants': .2, 'variables': .8},
    'functions': {'sin': .5, 'cos': .5},
    'operators': {'+': .8, '-': .2}
}



prior_1 = {
    'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
    'features': {'constants': .2, 'variables': .8},
    'functions': {'sin': .5, 'cos': .5},
    'operators': {'+': .5, '-': .5},
    'function_conditionals': {
        'sin': {
            'features': {'constants': 0., 'variables': 1.},
            'functions': {'sin': 0., 'cos': 1.},
            'operators': {'+': .5, '-': .5}
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

# import random
for _ in range(1, 5):
    burn(DEFAULT_PRIOR,
         _,
         "../src/equation_tree/data/_hashed_probabilities.json",
         10_000,
         0.5,
         )
    burn(
        prior_0,
        2,
        "../src/equation_tree/data/_hashed_probabilities.json",
        10_000,
        0.5,
    )
    burn(
        prior_1,
        2,
        "../src/equation_tree/data/_hashed_probabilities.json",
        10_000,
        0.5,
    )