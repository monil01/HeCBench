# CFD: full public 097K mesh, 2,000 local pseudo-time updates

The study uses the original five CUDA entry functions, 192 threads per block,
97,046 physical tetrahedral cells padded to 97,152, and an independent
192-cell analytic uniform-flow control. Each case executes the source's full
2,000 iterations: an old-state device copy, step coefficient calculation and
three flux/update stages. The stage coefficients are `1/(4-j)`; the source
comment does not establish a classical third-order Runge–Kutta method.

## Reference qualification and scientific contract

The original host padding loop assigns the last physical cell's normal back to
itself, leaving 1,272 padded normal components unwritten. The owned fixture loader
copies those components to the intended padded cells. Original CUDA source and
kernel bodies remain unchanged. This is a **conditional initialization repair**,
excluded from untouched-native aggregate comparisons. The CPU destination-set
witness makes no claim about observed values from an undefined GPU read.

The independent Float64 oracle constructs the Euler flux tensor directly and
checks every CPU integration stage for positive density and pressure. All GPU
paths check complete initialization, full state every 100 iterations and final
state/old state/flux/step buffers against this oracle. Acceptance is declared
before measurement: `abs(error) <= 2e-4 + 2e-3*abs(reference)`, all values finite,
positive density and pressure at captured states, exact guard/input/readonly
checks, exact padded-state copies and analytic freestream drift at most `3e-6`.
This engineering budget is not a physical-model or rigorous forward-error bound.

Actual pointer inputs are read back completely with 32 guards on both sides.
Float32 state and geometry and Int32 neighbors are encoded losslessly as Float64
in saved artifacts; artifact encoding does not change computation precision.
Source Float3 values become immutable three-component tuples in Julia.

The CFD model is compressible inviscid Euler with artificial numerical viscosity
and local pseudo-time coefficients. Pairwise face flux is antisymmetric. A
cell-varying pseudo-time coefficient does **not** imply conservation of the
unweighted global state sum per iteration. No lift/drag, mesh convergence,
physical time or application-level aerodynamic validation is claimed.

## Timing and observation adaptation

All four implementations use the same declared 42 timed segments: initialization
and twenty segments of 100 full iterations per case, each with whole-context
completion. Complete device readbacks and independent checks occur between
segments outside the timer. Every one of five trials is checked; only trial 1's
time is discarded and trials 2–5 are averaged. Extra fences, observations, host
launch overhead, cache effects and Julia host allocations remain covariates.
This is **not** the uninterrupted original CLI timing region.

## Clean-checkout fixture/oracle regeneration

Supply the public Rodinia 3.1 `fvcorr.domn.097K` file with SHA256
`43534e58454ba8baf95253e14d3b07150b9960577e922a954d6e51142a27abfa`.
This local asset comes from the public Rodinia dataset, not a SPEC copy. The
upstream locator is recorded in `reference_provenance.json`; no identity with the
unfetched HeCBench DVC archive is asserted. The 22 MB mesh and generated arrays
are intentionally omitted from Git and the compact paper.

From this directory:

```sh
python3 regenerate_fixture.py --mesh /path/to/fvcorr.domn.097K --threads 8
```

Requirements: Python standard library, GNU C++17 and OpenMP. The independent CPU
oracle runs all 2,000 iterations; exact generated SHA256/byte counts are in
`data_regeneration.json`. Regeneration must not run concurrently with an active
verification process reading these files. Saved artifacts can be independently
replayed with the standard-library command:

```sh
python3 audit_outputs.py --result /path/to/run/result.json --output audit.json
```

## Preserved failed controls

The connected 192/384-cell cropped domains, artificially closed with farfield
cuts, failed the independent CPU positive-pressure gate before any GPU execution
or target drafting. They are rejected controls, not verified CFD cases or
framework limitations. The first failed cell/stage was not captured; that gap is
explicit. The failed oracle candidate and logs are referenced under
`reference_history/`. Separate initial runner-asset staging and external-auditor
errors are instrumentation repairs, not CUDA/Julia device failures.

## Accepted evidence

All twenty full artifacts pass independent replay (`output_audit.json`), with
26 measured source/asset bindings unchanged. Final physical-cell conserved-state
and derived pressure errors are in `accuracy_summary.json`; these are separated
from errors in step coefficients and flux working arrays. Descriptive final flux
RMS and last-iteration changes do not establish a converged aerodynamic solution.
KA frontend candidate 1 rejects trailing `return`; candidate 2 passes. JACC
candidate 1 compiles but its mutable launch specification caches one block from
the small control and misses the full grid; captured cell 193 has density zero.
Explicit one/506-block specifications repair it. All five rejected captures remain
local; nonfinite diagnostic metrics caused the old strict-JSON receipt to fail,
so no result.json exists for that rejected run. No failed candidate was rerun just
to create a receipt. See `reference_history/` for the immutable candidates and
source-bound diagnoses.
