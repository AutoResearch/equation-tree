# Distance Metrics

To evaluate the precision of a symbolic regression algorithm, or the loss in training of such algorithm, we need a distance metric between equations. Evaluating such a distance is not straight forward. The Equation Tree pacakge therefore includes various metrics that can be used individually or in conjunction:

- [**Prediction Distance.**](#prediction-distance) Distance between function values as proposed by La Cava et al. (2021).
- [**Symbolic Solution.**](#symbolic-solution) Another metric proposed by La Cava et al. (2021) is called symbolic solution, designed to capture SR models that differ from the true model by a constant or scalar.
- [**Normalized Edit Distance.**](#normalized-edit-distance) Matsubara et al. (2022) propose a normalized edit distance for the trees. For a pair of two trees, edit distance computes the minimum cost to transform one to another with a sequence of operations, each of which either 1) inserts, 2) deletes, or 3) renames a node.

## Prediction Distance
coming soon ...

Pros
- ...

Cons
- Can be heavily reliant on the input sample it has been evaluated on

## Symbolic Solution
coming soon ...

Pros
- ...

Cons
- Is a binary if the equations do not only differ by a scalar or a constant

## Normalized Edit Distance
coming soon ...

Pros
- ...

Cons
- ...

## References

La Cava, W. G., Orzechowski, P., Burlacu, B., de França, F. O., Virgolin, M., Jin, Y., Kommenda, M., & Moore, J. H. "Contemporary Symbolic Regression Methods and their Relative Performance." In *CoRR* (2021), Available at: [https://arxiv.org/abs/2107.14351](https://arxiv.org/abs/2107.14351)

Matsubara, Y., Chiba, N., Igarashi, R., & Ushiku, Y. "Rethinking symbolic regression datasets and benchmarks for scientific discovery." In *arXiv preprint arXiv:2206.10540*. (2022), Available at: [https://arxiv.org/abs/2206.10540](https://arxiv.org/abs/2206.10540)