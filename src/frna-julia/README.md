# FRNA: RNA minimum-energy dynamic programming

The three Julia children implement the complete ten-stage GPU recurrence pipeline. The named experiment includes the original RD0260 77-base sequence and five 9–44-base controls. It does not run the larger HIV input, reconstruct a folding traceback, compute a partition function, or establish biological ground truth.

## Reference qualification

Original sources are intact in `../frna-cuda` and `reference_history`. Actual original diagnostics found 1,080 device uninitialized reads and a serial CPU initialization write past the allocation. `reference_provenance.json` binds the minimal terminal/WCA initialization and serial base-pointer corrections. The author table states `1.079*ln(n/30)`: original INT-scale100 code instead hard-coded the scale10 coefficient10.79. The configured parser retains the raw Float32 coefficient before multiplying by100; probes and an independent table-rule witness distinguish these conventions. All native GPU function bytes remain unchanged. This is a qualified corrected-reference study, excluded from untouched-native aggregates.

Four complete matrices and both boundary vectors are checked at initialization and every distance. Each of 20 artifacts contains 2,396,459 exact Float64-encoded integer fields, including 128,075 actual device input/initial-state/guard fields and 2,205,408 full source-CPU recurrence states. The source CPU recurrence is a crosscheck, not an independent folding algorithm. `check_small_structure_optima.py` independently enumerates every admissible noncrossing structure for five controls and evaluates their energy, establishing global optima under the declared table/filter conventions. The 77-base optimum lacks that independent exhaustive certificate. Original potential-neighbor pair filtering, finite INF14000, minimum loop3, internal-loop bound30, and integer rounding remain explicit.

## Reproduce from a clean checkout

From the repository root, with an unprivileged C++17 compiler and Python3:

```sh
python3 src/frna-julia/regenerate.py
python3 src/frna-julia/check_small_structure_optima.py --fixture-root src/frna-julia --output /tmp/frna_independent_optima.json
python3 src/frna-julia/make_config.py --output /tmp/frna_fourway.json
python3 tools/verify_portable_julia.py --config /tmp/frna_fourway.json
```

Regeneration downloads only hash-pinned original data archives, validates their exact sizes/MD5/SHA256 and all extracted regular files, compiles the owned CPU fixture generator, and verifies all five output hashes against `regeneration_contract.json`. It never executes downloaded code. Raw arrays are regenerable and need not be checked into Git. The data and larger HIV archive are recovered for provenance; the latter is not a measured input. To review fixture reproducibility without touching the measured input directory:

```sh
python3 src/frna-julia/regenerate.py --output-dir /tmp/frna_regenerated
```

The generated Julia recurrence files are committed. Optional source-statement regeneration needs `pycparser==2.22` in a user-space environment and `python3 src/frna-julia/generate_device_bodies.py --output /tmp/recurrence_bodies.jl`; it does not generate reference outputs. Package versions and environment file hashes are in per-model traits. Julia1.12.6/CUDA.jl6.2.1/KA0.9.45/JACC1.4.0 and native CUDA12.0.140 were measured on physical GPU6 V100. The central verifier serializes GPU work using the shared lock; do not wrap it in another lock.

## Timing and mapping

Every model performs an explicit complete warmup before five checked trials. Report the mean of trials2–5; all raw times are retained. The ROI is the sum of164 context-completed initialization/per-distance segments across six cases. Each distance includes all nine source stages; allocation/reset, full-state observations, and checks lie outside the ROI. Pauses between segments mean these are not unsegmented original CLI times. Mean seconds: native0.00855630125; CUDA.jl0.01010280025; KA0.01386403575; JACC0.02036160375. No hardware counter or roofline attribution was measured.

CUDA.jl uses the original scalar and256-lane shared reductions. KA owns ten kernels with framework local memory/barriers; JACC owns ten dispatch functions and explicitly uses CUDA static shared memory/block-barrier backend extensions. JACC dynamic shared memory is zero; static storage remains1KiB for reductions. KA/JACC flatten the source16×16 bulge block to256 physical lanes with identical logical tx/ty/reduction-index values. A shared-buffer closure barrier is explicit. Those mappings and host dispatch differences qualify performance comparisons.

## Evidence and cost

`verification_audit.json` independently hashes and checks every complete artifact against the input/state/guard layout. `audit_captures.py --result RESULT_JSON --output AUDIT_JSON` repeats this CPU-only review using retained local captures. `paper_observations.{json,md}`, `domain_snippet.json`, `implementation_traits.json`, and `effort_summary.json` record the RQ evidence and limits. Failed source candidates, compiler feedback and reference probes are retained separately. CUDA.jl required three program evaluations/two actual device-JIT evaluations; KA/JACC each passed their first. Setup, source-generation, oracle and instrumentation repairs are distinct units.

Actual usage counters are prospective translation-window measurements, including repeated/cached inherited context and coordination. Cached input is a subset of input; reasoning is a subset of output. Native preparation and verification/annotation have separate windows. KA/JACC reuse the accepted mathematical helpers and harness: measured costs describe this ordered marginal workflow, not independent cold-start abstraction costs or reconstructed historical consumption.
