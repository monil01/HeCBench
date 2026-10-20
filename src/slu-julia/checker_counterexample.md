# Original checksum gate: CPU-only counterexample

This is an executed counterfactual against the untouched original
`Symbolic_Matrix::ABFTCheckResult`, **not an observed incorrect GPU result**.

For original matrix `A=I`, choose combined factors encoding

```
L = [1 0; 1 1]        U = [0.5 0; 0 1]
L*U = [0.5 0; 0.5 1]
```

Both columns retain the original sum of one. The original source checksum gate
prints `Results passed ABFT check.` even though the full product differs from A.
The independently executed full-product gate rejects it: maximum normalized
componentwise backward error is 1 and normwise backward error is 0.5, against the
declared 1e-4 budgets. `checker_counterexample.json` contains actual output.

The supplied `x=[1,1]` satisfies the original identity system, deliberately isolating
factor verification: passing a solve residual alone also cannot validate all factors.
Neither this witness nor the separate racecheck warning proves an actual wrong
original GPU output. Both motivate keeping the independent complete-factor gate
alongside source self-checks and numerical comparisons.

Reproduce on CPU with existing g++ and the extracted original NICSLU headers:

```sh
g++ -std=c++17 -O3 -fopenmp -DVERIFY \
  -I src/slu-julia/reference_history/include \
  -I /tmp/hecbench-stencil-ports-slu-original/src/nicslu/include \
  src/slu-julia/checker_counterexample.cpp \
  src/slu-julia/reference_history/src/symbolic.cpp -o /tmp/slu-checker-witness
/tmp/slu-checker-witness
```
