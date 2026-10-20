# Remaining executor/build gates

CPU-only review. No compiler, make, benchmark, GPU, sudo or Git command was run.
No source, accepted result, shared cache, existing helper or shared ledger changed.

## Original thirteen executor proposals

`existing_thirteen_executor_coverage.json` binds the existing six-case helper and
v4 overlay inventory. XLQC4, Pad4, Leukocyte4 and IntervalCUDA already have explicit
zero-gate plan proposals there. This is not build or runtime evidence. Their
precreation/fixture cloning/cache replacement semantics must remain case-specific;
none may be replaced by a blanket gate waiver.

## Four Halo-finder source reviews

`approved_halo_source_reviews.json` is root-planner compatible and binds current
result/config/complete execution sources and line evidence. The literal MPS path
is inside disabled preprocessor code; USE_SERIAL_COSMO selects rank0/oneprocess.
The builder uses private TemporaryDirectory intermediates and explicit relative
output, input fixtures are read-only, and trial files are cwd-relative.

Root must enforce CUDA_VISIBLE_DEVICES=6, HACC_NUM_CUDA_DEV unset or1, and private
TMPDIR. Default logicaldevice0 then refers to GPU6; no MPS or MPI process/context
change is authorized. The review did not build, preprocess or execute the program.

## Separate SLU/SVD helper: seven plan proposals

`tools/v100_roofline_prepare_extra.py` owns only exact SLU and SVD3x3 cases:

- SLU4 compiles the same NICSLU Makefile object list with the original GCC PIC/g/
  O2/msse2/NO_ATOMIC/SSE2 flags, but into private scratch objects/archives. The
  old cache builder and global build.lock/manifest are never invoked. Original
  nvcc flags/source/CPU controller/GPU functions remain. Fixture matrix paths and
  canonical prefixes are bound/read-only; prepare=0 is required. Private captures
  redirect to scratch and x.dat/output_trialN.bin remain cwd-relative.
- SVD3 targets redirect g++ checker output and actual runtime CHECK_LIB to scratch.
  Exact copied common/entry include patches preserve canonical input/device-source
  lookups. Remove only the inline historical STUDY_RAW_DIR env prefix and use a
  fresh private_raw environment. Same Julia executable/project, kernels, solver
  iterations and mathematical checks remain; property agreement is not raw
  singular-vector identity. Original study_check.so is never overwritten/loaded.

`proposed_extra_inventory/summary.json` contains seven ready *input/build plans*,
not executed builds or rooflines. All original execution bindings were rehashed.
The default helper writes plans/manifests/patches only under a new /tmp output root;
`--stage` is optional copying, never building. `--build-manifest` is a hash-pinned
compiler/archive-only entrypoint reserved for root/collector execution. It checks
inputs before/after, requires scratch cwd and absent outputs, confines compiler
outputs to scratch, and never executes a target. No such invocation occurred here.

Example new isolated inventory (do not reuse the already-created output root):

```bash
python3 tools/v100_roofline_prepare_extra.py \
 --base-plan-root /tmp/roofline-method-review/reviewed_plan_v4_cap512 \
 --output-root /tmp/NEW_EXCLUSIVE_EXTRA_INVENTORY
```

Add `--stage` only after root review/storage budgeting. Staged asset budget and
free-space reserve are explicit. Plans retain ROI/replay/counter mapping unknown;
no preparation readiness creates a scientific or whole-ROI profiling claim.
