# SSS: full source search with retained scientific host controller

This study restores the supplied `f9_n150_p50` (150 observations, 50 variables)
complete seeded stochastic shotgun search. The original scientific C++/GSL
controller is retained through a common host C ABI. Each translated implementation
must provide its own two GPU clique predicates; this is **not a pure Julia port of
the scientific CPU controller**. All four implementations passed the declared named contract; see
`restoration_checkpoint.json`, `final_saved_full_audit.json` and the selected immutable result receipts.

## Contract

The native preparation checked 12,469,275 actual GPU add/delete-edge decisions
against an independently expressed maximum-cardinality-search chordality
criterion, including the exact unique-clique index required for deletion scores.
The five-trial native reference also checks every inactive output slot, complete
actual device packed inputs, 256 physical redzone words, all finite search states,
partition labels, and symmetric binary graph adjacency. Full search traces,
source model lists and actual text result files are retained. Exact all-four
comparison uses every byte of a lossless typed capture, **not checksums as a
replacement for numerical values**. `uint8` describes the serialization, not the
Int16 GPU or Float32 scientific computation.

The independent graph check is not an independent implementation of G-Wishart
integration or a Bayesian posterior/inference-quality certificate. The six-vertex
exhaustive CPU graph witness is supplemental; the primary instrumented workflow
checks every actual 50-vertex GPU proposal. The original five CLI completions had
an absent `RES` directory and unchecked writers; they do not establish complete
numerical verification. The new harness creates and retains the result files.

## Qualification and timing

Original source files and the two native CUDA bodies are preserved. Minimal host
changes define unused packed clique/separator tails, avoid an unused no-change
index read, initialize source device buffers and add an explicit input-ready
cross-stream event. These are disclosed definedness/ordering qualifications,
not evidence of an observed original scientific error. This contract is excluded
from untouched-native aggregate comparisons.

The common ROI is the full original search loop with ordered device-to-device
input/output tracing and host state observations included. Whole-device waits
bound the ROI. Initialization, final readback/writing and the independent graph
check are outside it. One complete warmup precedes five independently executed
checked trials; report the mean of trials 2–5. All implementations retain the
same scientific C++ controller and library. Julia callback/wrapper overhead,
source host allocation and cooperative synchronization/resource differences are
covariates; these are not kernel-only or isolated frontend-effect measurements.

CUDA.jl uses its own cooperative barriers in place of implicit source warp
communication. The final Boolean path result has one writer after a cooperative
OR, preserving the source observable decision while avoiding redundant stores.
The original native body is unchanged, and source synchronization hazards are
not called observed defects merely because the translation uses explicit barriers.

## Evidence and storage

`native_diagnostics/native_five_trial_receipt.json` records the selected native
reference. `native_diagnostics/candidate_history/preparation01` preserves the
older preparation without inactive-tail capture. It was preparation, not a
competing retained timing series. Each final capture is 242,952,380 bytes; the
first 181,368,048 bytes contain source data and every actual GPU packed input.
The full typed layout and graph-record sidecars permit lossless inspection.

New captures are on `/tmp` with canonical result aliases and explicit storage
provenance. Independently created, completely hash- and byte-verified identical
owned files may share an inode; every logical path and byte is retained, with
creation-stat and consolidation proof. These local captures require preservation
before reboot. No historical evidence was deleted to make space.

Cold rebuilding uses `python3 src/sss-julia/prepare_study.py` then
`python3 src/sss-julia/build_study.py BUILD_DIRECTORY`. The original data/init1,
original source archive and user-space GSL 2.8 provenance are bound in each
configuration. No OS updates, privileged installation or sudo were used.


## Frozen result and decoded scope

Selected all-four result: `20261020T154113208424Z_sss`. Mean trials 2–5 in
seconds: native 34.5383132815, CUDA.jl 34.01827727975, KA 34.4115183195,
JACC 34.712893722. All 20 full captures are exact. CPU-only saved-data audit
checked all 4,859,047,600 logical bytes, 163 source/dependency bindings, every
source writer/graph sidecar, and the complete typed state layout. One full typed
mathematical decode was reused only after full byte comparisons of all captures.
It checked 2,288 states, 6,460 distinct chordal graphs and all 5,099 writer fields.

The compiled source GPU branch stops at the first completed mode: it increments
`nmodes` to two, resets `k` to zero, and breaks before the alternate restart path.
There are 2,285 observed iteration states plus final states. This original policy
is retained; nominal `maxNmodes=2` does not mean two observed completed modes.
The first additional CPU decoder expected a nonreset final counter and failed;
its exact source/log is preserved. Correcting that decoder required no GPU rerun,
measured source change, tolerance change or altered primary acceptance.

CUDA.jl had two host binding errors before device JIT; KA and JACC passed their
first complete candidate. Each has two authored device bodies and one successful
device-JIT process. `effort_accounting.json` retains observed phase counters with
cached/reasoning subsets, inherited/context reuse and cross-planning caveats.
All SSS token windows are excluded from forecast calibration: these are actual
inclusive activity windows, not clean independent prices per frontend.

KA has fixed 180-word local workspace plus 12 flags. JACC uses explicit 8/384-byte
dynamic shared allocations; its readonly preload parents (including 68 physical
guard words) have complete actual pre/post readbacks. These implementation inputs
are separate from common scientific problem inputs. Native/CUDA variable dynamic
workspace, KA fixed storage, JACC preloads, wrapper overhead and retained host
work remain covariates; no GPU fraction or performance cause was measured.

## Cold preparation

Use Python 3.11+ (with safe tar extraction support), Julia 1.12.6 and the pinned
CUDA/KA/JACC project manifests on the declared CUDA12.0/sm70 platform. Scientific
source data/init files are the small original assets in `original_native/DATA`.
No raw GPU output is a required source input.

1. `python3 src/sss-julia/fetch_build_gsl.py` fetches the exact SHA256-pinned GSL2.8
   source and builds only its explicit user cache/prefix; it changes no OS packages.
2. Instantiate the pinned Julia environments if necessary.
3. `python3 src/sss-julia/prepare_cold_protocol.py --build NEW_BUILD_DIR --captures NEW_CAPTURE_ROOT --out-config NEW_CONFIG.json --julia PATH_TO_JULIA`
   compiles a new common host/native binary and prepares a fresh all-four config.
   It executes no GPU series. New paths must not overwrite existing evidence.
4. Review storage/hardware/environment qualification, then use the central
   verifier on that prepared config. The verifier owns the GPU6 lock; do not add
   an outer lock. Preserve its new captures and receipts separately.

These cold helpers were source-reviewed and prepared, not claimed to be a second
executed all-four reproduction. The actual selected build and all original
process/capture evidence remain preserved. Different toolchains/hardware require
new declared strata and are not silently comparable to this result.
