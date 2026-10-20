# Halo-finder: qualified RCB particle-gravity pipeline

This source folder evaluates gravitational particle forces; it does not assign halo labels. The named experiment retains the complete source C++ RCB tree construction, interaction-list traversal, transfers and GPU force evaluation. It uses the original first10,001-particle seed12 sphere input and31/65-particle controls, with opening angle zero (the direct-interaction limit). The first large input is byte-identical to the original source generator across70,007 Float32 fields.

## Scientific qualification

The original CUDA kernel ignores the supplied `ForceLaw` and evaluates a softened, truncated inverse-distance force. The original CLI `N` comparator evaluates unsoftened Newton gravity, while `S` subtracts a grid-force term absent from the GPU code. An unchanged-kernel one-pair test returns3.77146435 versus Newton4, matching the independent softened expectation3.77146412. These original API/model defects remain flagged; no original N/S fidelity is claimed and these results are excluded from unqualified scientific-fidelity/untouched-native aggregates.

The **actual named softened-gravity operator** is independently checked at every particle using Float64 all-pairs calculations. Across20 captures, relative L2 error is3.999e-8–8.455e-8; every component satisfies the predeclared abs0.05+rel2e-4 engineering budget. That combined contract is not a uniform relative-error guarantee near canceled components. Net-force and torque residuals are checked as central-force invariants. No halo catalog, global cosmological solution, orbital integration, nonunit-mass accuracy, cutoff-edge behavior or other multipole opening angles are certified.

An original two-particle full-tree CLI control raises SIGFPE in `printStats` before force evaluation. It remains an unverified auxiliary scope; the direct one-pair GPU test is a separate successful kernel witness. Regenerating its mathematical oracle does not certify that full-tree path.

## Complete inputs and outputs

Each trial contains22,436,535 Float64-encoded fields, including70,679 exact initial Float32 bit words. Every source leaf uses an auxiliary GPU copy kernel to read all actual GPU-facing input spans; it performs no force calculations. Coordinates/masses are checked against the complete captured global bit patterns and encoded losslessly as two-byte particle IDs. Every dynamic list must contain every global particle exactly once; every target appears exactly once. Input/metadata byte fields are strictly identical across all20 captures, independent of output tolerance.

All leaf force outputs and final global force fields are retained, not just the original test-particle norm. The independent audit makes1,211,640 full force comparisons, validates all input bytes and saved hashes, and checks net force/torque. Guards are checked exactly in the bound C source before/after each launch and on free; guard bytes are not redundantly serialized into the numerical artifact. Read-only GPU inputs are checked by actual GPU echo before/after evaluation.

## Memory and implementation boundaries

The source calls `cudaMallocHost` for GPU-facing force buffers. Pointer attributes confirm host memory with equal host/device aliases; the pinned-host/UVA mapping is preserved for every model. Non-owning CuArray views do not convert these arrays to HBM storage. No measured bandwidth or roofline attribution is made.

Every target authors the complete scientific GPU force kernel. The source C++ tree/list/allocator/controller is shared through FFI. The nonnull target callback performs Julia GPU dispatch; the compiled native scientific fallback is not used by a target. A shared native copy-only verification kernel runs outside the ROI. This is a hybrid host pipeline, not a pure whole-program Julia rewrite.

All mappings retain4×4 tiles, cached vector/scalar reads, correct-rounded Float32 reciprocal square root, full32-lane shuffle sums and Float32 atomics through explicit CUDA backend extensions. KA physically flattens blocks/threads while Cartesian indices retain the logical32×4 mapping and warp lanes. CUDA.jl mirrors source128-thread/min7-block launch hints; KA exposes maxthreads/always_inline but not the min7 hint, and the selected JACC builder does not forward those global compiler hints. These are selected-path control differences, not proofs that the frameworks cannot express the numerical operator.

## Reproduce

From the repository root, using unprivileged C++17/OpenMP/CUDA tools, Python3 and the pinned Julia projects:

```sh
python3 src/halo-finder-julia/regenerate.py
python3 src/halo-finder-julia/make_config.py --output /tmp/halo_fourway.json
python3 tools/verify_portable_julia.py --config /tmp/halo_fourway.json
```

The fixture generator requires no GPU or external data. It reproduces all declared input/oracle hashes and fails closed on incompatible compiler/libm/POSIX RNG behavior. It was independently exercised into `/tmp/halo_fixture_regeneration_review`. `prepare_tree_adapter.py` reproduces only the committed allocation/dispatch/observer hooks; mathematical CPU tree and GPU bodies remain unchanged. The original complete sources and license notices remain under `reference_history` and `../halo-finder-cuda`.

The central verifier owns the GPU6 lock; do not wrap it in another lock. Selected data use serial source flags, not a multi-rank/MPI performance experiment.

The independent CPU audit uses a private NumPy2.2.6 target, without changing the OS or shared verifier:

```sh
python3 -m pip install --target /tmp/halo_audit_numpy --only-binary=:all: numpy==2.2.6
PYTHONPATH=/tmp/halo_audit_numpy python3 src/halo-finder-julia/audit_captures.py --result RESULT_JSON --output AUDIT_JSON
```

`--run-root` can relocate retained run folders. The primary results used the central verifier’s original Python struct comparison engine; this optional analysis dependency does not retroactively change that evidence.

## Timings, attempts and actual tokens

A complete pipeline warmup precedes five checked trials. Ignore only timing1 and average timings2–5. Means: native0.717691658; CUDA.jl0.7572313355; KA0.76398400825; JACC0.7116964065 seconds. All raw trial times are preserved. The ROI covers complete RCB construction/list work, pinned workspace allocation, transfers, force dispatch/completion and destruction. Explicit input echo/observation/check intervals are excluded by measured interval subtraction; global fixture/setup is outside the ROI. This is not uninterrupted CLI time or kernel-only performance, and no causal explanation is inferred from these small differences.

The first CUDA.jl candidate compiled but exhausted dynamic GPU memory through private arrays, then the independent force gate rejected zero output. Inlining generated helpers and applying source force-inline behavior repaired lowering without changing equations. CUDA.jl used two program/JIT evaluations; KA/JACC passed their first after reusing that feedback. The original failed source and165MB runtime log remain local; a compact scientific excerpt is archived. Native adapter/link/copy-kernel integration failures and small-N source diagnostics have separate provenance and units.

Prospective actual token counters record input/cached-input/output/reasoning with model context and marker lag. Cached input/reasoning are subsets, not extra totals. Native preparation, translation, verification and annotation have separate windows. Counts include inherited context, feedback, complete verification and coordination; KA/JACC reuse accepted helpers. They describe this ordered marginal workflow, not independent cold-start framework cost or reconstructed historical consumption.
