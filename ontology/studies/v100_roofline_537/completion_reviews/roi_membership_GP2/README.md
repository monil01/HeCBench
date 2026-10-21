# Additional G–P source membership proposals

Fresh private packet, same normalized schema as GP1: `parent_reviews`, `selected_capture_bindings`, `sources`, `patches` with record/stage canonical hashes. No shared mutation. Current parent gzip/raw review bytes were rebound after root's prior enrichment, not reused from the old GP1 snapshot.

## Selected scope

Four **new** names, eleven currently successful selected model records:

| Name / scientific stage | Reviewed models | Withheld models |
|---|---|---|
| laplace3d / 3D relaxation stencil, fixed twenty steps | CUDA, CUDA.jl, KA, JACC | none |
| particle-diffusion / position update and private UInt64 visit counts | all four | none |
| haccmk / particle force polynomial and velocity update | CUDA, CUDA.jl | KA/JACC selected profiling processes failed |
| jacobian / softmax full-Jacobian materialization | KA | CUDA/CUDA.jl/JACC selected profiling processes failed |

Source precision and algorithm remain the original qualified named contracts. A source inclusion proof does not certify all arithmetic is counted by scalar FP metrics (e.g. integer visit counts, square-root/exp work). This annotation does not reinterpret the scientific family or introduce a performance cause.

## Proposed stage updates

Eighteen existing stages: eleven timed-family inclusion proofs, one separate native warmup exclusion, six dormant legacy CLI callsite exclusions. Dormant code is explicitly `callsite_kind:dormant`; it is not claimed to execute during setup. Exclusion uses the existing allowed source_setup_exclusion_proved status and disables that candidate regex to prevent duplicate device-family matching. Every stage retains its original canonical hash and exact callsite.

- **laplace3d:** selected native study's repeat dispatch is enclosed by steady_clock start/synchronized elapsed; earlier launch is warmup; renamed historical_main sites never execute. Julia study_prepare returns a named closure invoked by common twenty-step timed loop after reset. JACC explicitly passes laplace_kernel! to parallel_for; KA constructs named kernel. Two input fixtures remain distinct, no complete convergence claim.
- **particle-diffusion:** native launch lambda is called after reset and timer start, then synchronized; same lambda also warms up before trials. Imported legacy main is renamed. Julia study_prepare explicitly binds simulation_kernel!/portable_particle!, invoked inside each timed repeat; random arrays, resets and UInt64 fill are outside. CUDAjl study strips historical exit(main(ARGS)). Generic fill observations remain unassigned; no broad setup mapping is inferred.
- **haccmk:** native lambda is timed; imported native main dormant. CUDAjl factory closure is passed through hacc_study to OwnedStudy.run_trials!, whose repeat loop is enclosed by start/time completion. Velocity reset/readback are outside; included main PROGRAM_FILE guard prevents legacy main execution. Own loop has one source-configured update, not a full cosmological simulation.
- **jacobian:** actual selected argv `k2 512` selects softmax_jacobian_ka2! in factory; its returned kernel closure reaches OwnedStudy's timed repeat via jacobian_study. The k1 candidate is **left unchanged/unknown**, not falsely labeled setup, because it is a parameter-inactive implementation. Only k2 membership is promoted, and source mode selection is explicit in proof and actual target argv.

## Evidence and merge

Nineteen proof source files match actual selected frozen profile-plan execution bindings by resolved absolute path and SHA. Exact source snippets/callsite/timer lines, selected result/configuration/manifest/derived SHA, current parent gzip/raw SHA and record/stage indices/canonical SHA are included. Selected observed identities match at most one proposed candidate family. This is unique source-family matching, **not observed timed phase**.

Root must revalidate all bindings and parent stage hashes, preserve historical compressed review bytes, and apply listed updates within the existing record authority. Emit no second overlapping `(benchmark,model)` review. If another worker/root changes a shared parent review first, compressed/raw SHA drift requires explicit rebase, not silent application.

All original coverage/steady-state/launch-census/bottleneck qualifications remain. No runtime first-sample phase, complete family or elapsed ROI aggregate, causal assignment, or launch-multiplicity extrapolation is proposed. All phase-independent fields on existing observations should stay unchanged. No GPU query/run, build, test, Git, source/tool/ledger/paper edit occurred. Shared annotation usage receipt is separate from translation costs.
