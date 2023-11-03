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