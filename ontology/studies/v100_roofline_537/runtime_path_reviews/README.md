# Remaining runtime paths: CPU-only review

No builds, staging, target execution, GPU jobs, sudo, Git, source edits or shared
ledger mutations occurred. These files are proposals for root integration.

## Seventeen narrowly scoped planner-compatible approvals

`approved_runtime_path_reviews.json` has `models` entries compatible with the
existing preparation review format: exact result/config SHA, unchanged complete
execution bindings, acknowledged current gate, output-isolation attestation,
exact argv replacements and source-line/hash evidence. All current execution
bindings were rehashed; no mismatch was waived.

- columnarSolver CUDA.jl/KA/JACC: redirect only final private capture directory;
  CPU gold stays a read-only first argument and primary files stay cwd-relative.
- easyWave CUDA: the flagged root only constructs read-only input/oracle streams;
  outputs use STUDY_OUTPUT_DIR. No absolute-input rewrite is proposed.
- gibbs JACC: final scratch trace directory only; library argument unchanged.
- grep all four: private NFA-graph observer outputs only; preserve full observer.
- heartwall all four: private full-frame replay output only; inputs/checker intact.
- sssp all four: final private state root only; graph/oracle and hybrid controller
  arguments unchanged. This is not a replay-safety or whole-ROI approval.

Root must merge/review these with existing immutable reviews and regenerate new
plans. Old plans/results remain unchanged; no profiling was performed here.

## Six executable-role proofs are NOT full isolation approvals

`nested_executable_role_reviews.json` proves the bound Julia path is an executable
argument, with its resolved file hash, for minibude and Sparkler (three versions
each). It explicitly grants no gate waivers and sets output isolation false:

- minibude's command begins `env STUDY_RAW_DIR=<historical> julia ...`. Merely
  setting process environment does not override this inline assignment. The new
  plan must remove only that env wrapper/assignment, retain the same executable
  and remaining argv, and set STUDY_RAW_DIR to fresh scratch. All private writes
  are source-proven descendants of that variable; assets remain source-read-only.
- Sparkler's MPI wrapper executes the supplied Julia command on four source ranks,
  all sharing GPU6; `--target-processes all` is required. Its Julia common source
  constructs the historical sparkler_primary CAS/ledger root. Use a hash-bound
  source overlay redirecting this output root, retain canonical source ROOT and
  `LIB=joinpath(pwd(),"study_backend.so")`, and keep MPI/concurrency semantics.
  Rank/phase coverage and replay suitability remain separate reviews.

## Eight explicit new-plan proposals, not executor approvals

`new_plan_proposals.json` contains those six slots plus:

- columnarSolver CUDA: source wrapper creates fixed /tmp/columnarSolver_native/
  checked_PID and executes an old fixed binary. Use a fresh wrapper output root,
  newly built scratch executable and unchanged read-only CPU gold. A new plan must
  declare the compiler command and exact source/data bindings; no old binary or
  historical capture may be modified.
- stencil1d CUDA: mkstemp already creates exclusively, so no overwrite was observed.
  It is nevertheless outside designated scratch. A source-only filename-template
  overlay can retain mkstemp semantics and complete output under fresh scratch;
  existing temporary evidence must not be deleted. The original gate remains.

Source snippets can include long generated lines; their complete file SHA binds
uncropped source. Output isolation does not prove scientific correctness, timing
membership, supported replay, absence of runtime/JIT caches or device readiness.
