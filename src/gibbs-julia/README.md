# GIBBS: scoped Poisson–Gamma adaptive MCMC

The named primary runs 513 deterministic observations and 32 complete GPU/CPU feedback steps. Actual unchanged-source CLI readiness was checked both on this fixture and on the recovered original DVC dataset (3,055 observations, 1,000 steps). Full original-dataset numerical trajectory and posterior convergence are not certified.

## Reproduce

From repository root, with pinned CUDA 12.0 / sm_70, Julia projects and Python 3.12:

```sh
python3 src/gibbs-julia/generate_assets.py
python3 src/gibbs-julia/build_bridge.py
python3 src/gibbs-julia/prepare_cold_config.py /tmp/gibbs_cold.json
python3 tools/verify_portable_julia.py gibbs --config /tmp/gibbs_cold.json
python3 src/gibbs-julia/audit_field_contract.py PATH_TO_NEW_RESULT_JSON /tmp/gibbs_field_audit.json
```

Acceptance requires the last command's complete field audit, including all four models and all five trials; the central 2e-4 envelope alone is insufficient. Theta/a,b retain strict3e-5 absolute error; reductions2e-4; actual inputs, raw RNG states and guards exact. The committed reviewed study plan is preserved by the fixture generator. No saved GPU capture substitutes for an independently evaluated draw.

## Mechanism and scope

Each target authors XORWOW/Box–Muller/Marsaglia conditional sampling and shared-memory reductions. Original native setup_kernel/curand_init executes via an explicit C ABI before timing; source CPU adaptive a/b helpers execute via C ABI between timed GPU fragments. This is a mixed host-control/native-initialization integration, not a completely portable application. The device sampling/reduction bodies use own framework APIs. Native source is unchanged; only resettable host sigma/seed across repeated study trials is an instrumentation adaptation.

All five trials preserve every theta, block partial, RNG state and parameter at every step, with actual device input/guard readbacks. Each time is the sum of 32 two-kernel launch/completion intervals; it excludes CPU feedback and captures, as in the source timer. The first time is discarded and the remaining four averaged. JACC uses an explicit 2048 B shared template/preload; source/CUDA.jl/KA use static scratch.

The initial unfused CPU oracle rejection, isolated FMA diagnostic, original uniform-tolerance mismatch and all old series are preserved. No full-theta error budget was relaxed or stochastic mismatch canonicalized. Read `study_adjudications.json`, `native_readiness.json`, `scientific_witness.json` and `paper_observations.md`. Essential CPU binaries regenerate exactly; raw GPU captures and compiled shared objects remain local. Cold builds require external pinned vendor headers, not copied into this source packet.
