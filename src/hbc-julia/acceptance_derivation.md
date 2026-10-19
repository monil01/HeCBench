# Pre-GPU componentwise acceptance derivation

This contract is written after independent CPU source/input inspection and before any native or Julia GPU trial. The full-data long-double oracle finds maximum shortest-path count 32,083,586 and eccentricity 22; UInt64 counts cannot overflow for these fixtures, and every count is exactly encodable in Float64 capture. Casting counts to Float32 can round, so that rounding is included.

Let u=2^-24. All dependency terms are nonnegative. Every nonzero shortest-path count is at least one. The smallest possible ratio in the measured graphs is greater than 1/32,083,586, well above Float32 underflow. Distances/counts remain integers. The maximum dependency/centrality values are far below Float32 overflow. Optional source approximation is outside this contract.

For successor edge v→w at the next BFS depth, the mathematical contribution is

`t(v,w) = sigma[v]/sigma[w] * (1 + delta[w])`.

Independent long-double Brandes computes the mathematical dependencies. If E[w] bounds the absolute Float32 dependency error, a conservative term upper bound is

`F * sigma[v]/sigma[w] * (1 + delta[w] + E[w])`,

where `F=(1+u)^4/(1-u)` covers two integer-to-Float32 conversions, division, the addition of one, and multiplication. This deliberately allows numerator rounding upward and denominator rounding downward. FMA contraction cannot exceed this separate-operation envelope.

For r successor contributions, their summation order can be arbitrary. With `gamma(r)=r*u/(1-r*u)`, propagate

`E[v] = sum(term_upper - t) + gamma(r) * (sum(t) + sum(term_upper - t))`.

Backward depth order makes this a finite positive recurrence; no cancellation assumption is needed. The source's omitted root dependency is exactly zero in device storage and excluded from centrality. For vertex v, let p be the number of strictly positive contributions across sources. Its final centrality bound, including arbitrary atomic accumulation order, is

`B[v] = (sum_sources E[v] + gamma(p)*(sum_sources delta[v] + sum_sources E[v]))/2`.

Division by two is exact for these normal binary values. The CPU oracle writes each mathematical BC value and B[v] to `oracle.bin` before GPU execution. A small 1e-9 absolute allowance covers the long-double-to-Float64 oracle representation; zero mathematical BC must still be exactly zero. The path33 and star1057 fixtures have a stronger exact closed-form requirement: path vertex i has i*(32-i), and the star has hub 557040 and every leaf zero. Their integer intermediates remain exactly representable in Float32.

Each model must pass this per-vertex independent criterion. Cross-model comparison uses the largest precomputed `2*b/(1-b)` where b is the per-case maximum B[v]/BC[v], plus the declared 1e-9 absolute allowance. This follows from bounding two independent computations around the same positive mathematical result; it is not a post-observation tolerance change. The central comparison is broader than some per-vertex independent bounds, which remain mandatory. Report measured absolute/relative oracle errors separately; a budget is not an observed error or proof of arbitrary-input accuracy.

The global degree/depth `positive_rounding_chain_K` in the stats is an additional looser source-depth bound retained for audit; the final budgets use the componentwise recurrence above. The first global-only CPU oracle candidate is archived. No GPU data informed either derivation.
