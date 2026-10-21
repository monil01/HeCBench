# Five-name scientific ROI membership proposal — root review only

`proposed_stage_patches.json` uses the current GP2 root-merge schema and contains
**52 timed-family inclusions,4 setup exclusions and17 duplicate-family selector
nullifications across20 selected completed model records**. The five names are
all-pairs-distance,amgmk,asta,adv,bh. No shared authority, source, ledger, paper or
previously adopted25-name record was edited. GPU/build/import/test count0.

## Snapshot and merge prerequisites

The preparation opened the existing `/tmp/hecbench-roofline-metadata.lock` in
read-only mode and took LOCK_SH for the initial selected-ledger/stage-document/
finished-manifest/member snapshot, releasing it before source analysis. A second
short LOCK_SH snapshot confirmed every saved input byte unchanged. Current source
bytes were independently matched to actual captured plan/additional bindings and
rehashed after analysis. There was no metadata or source drift. These are two
bounded coherent read snapshots, not an uninterrupted lock over analysis.

Private input_snapshot* folders retain original compressed review/members, ledgers,
manifests and config/result contracts. The earlier private v1 packet/script are
preserved; final active packet adds arithmetic/branch/atomic line evidence. No
accepted measurement or scientific checking procedure was rerun.

Each record includes exact parent compressed/raw SHA, record canonical SHA, stage
canonical SHA, selected plan/provenance/derivation/manifest hashes, config/result
hashes/CID and source SHA+line excerpts. Root must take a new LOCK_EX and recheck
all selected parents and stage/preimage hashes before merging. Hold on drift.
The unchanged record-level full_ROI/steady_state/multiplicity/causal qualifiers
remain preserved. No actual sample-phase field is promoted by this packet.

## Case-specific boundaries

- **all-pairs-distance (4 inclusions,2 aliases).** The named224×4096 contract invokes
  nativek3 only, inside the launch timer after reset. Julia factory closures run
  inside OwnedStudy.run_trials!. Inputs are categorical bytes; outputs count
  unequal symbols in Int32. Nativek3 uses integer CUB block reduction/lane0store;
  Julia maps one sequential workitem per pair, without atomics. Nativek1/k2 remain
  distinct unselected candidates and are not nulled as aliases or falsely called
  setup. Output reset, CPU oracle/readback and artifact creation are outside ROI.
- **amgmk (4 inclusions,4 exclusions,3 aliases).** Float64 in-place asynchronous
  relaxation is the only timed GPU family. Primary elapsed sum also contains CPU
  matvec and CPUaxpy, so this family is not the whole primary ROI. Int32 parameter
  observer is launched/read back **before** the clocks and is excluded as setup.
  Device resets, residual/property checks and raw captures are outside timing;
  equilibrium/warmup uses the same relax family. Property-equivalence, scheduling
  variability and no identical-raw-state claim remain unchanged. CUDA.jl's two
  ambiguous union regexes are refined into distinct relax vs observe functions.
- **asta (4 inclusions,1 alias).** This source is an atomic cycle-following in-place
  transpose. Float32 values are payload copies, not floating arithmetic; Int32
  head/finished atomics, integer permutation/cycle indices and barriers control it.
  Each one!/native one callback times only transpose+completion; input/reset fills,
  output/flag/head checks and files are outside. KA/JACC use explicit CUDA device
  atomics, and JACC's8B shared control is a resource covariate. Same family warms up.
- **adv (4 inclusions,2 aliases).** NativeFloat64 cubature advection uses one fused
  native family; each Julia timed closure invokes all seven Val-specialized
  interpolation/derivative/flux/backprojection stages. Selected stage bounds and
  integer indexing are source-bound; no atomic operation is in this stage body.
  Allocation/upload/workspace setup/warmup precedes the timer, correctness/readback
  follows; overwrite behavior means no per-repeat reset inside timed closure.
- **bh (36 inclusions,9 aliases).** All nine source tree-pipeline families belong
  inside run: init, bbox,clear1,tree-build,clear2,summarize,sort,force,integrate.
  Float32 physics coexists with integer topology/locks/counters, CUDA atomics,
  readiness/volatile loads, warp votes, barriers and rsqrt. Source force-opening
  predicates and step>0 correction remain; none is a causal counter diagnosis.
  External state reset and full-state/topology/guard/oracle audits are outside;
  internal init/clears/tree construction are timed. Five scoped case selections
  remain explicit; no dispatch-count/size/rank extrapolation is introduced.

## Retained AMG overlay source

The actual profile-only AMG copies were disposable and no longer exist. Their
**complete exact text** is retained in the selected immutable plan's
`overlay_provenance.profile_source_patches[].text`, and its SHA matches both the
captured copySHA and execution bindings. This packet carries that exact text,
original/copy paths/SHAs and selected-plan hash in
`captured_overlay_source_evidence`. It does not recreate source at historical
scratch paths or claim those files survived. Root should verify this retained-text
join when merging, in addition to the ordinary original-source hash checks.
The original source bytes are themselves captured before/after; output/include
path-only changes preserve the inspected timer/callback/arithmetic sequence.

## Selector aliases and nonclaims

Alias-only changes set model_kernel_regex:null and kernel_family_alias_of. They
preserve all old metadata and do not assert that a real warmup call was timed.
Every canonical pattern is a single exact function-family identity, and all saved
kernel observations have at most one resulting stage match. Remaining unknown
runtime fill/library kernels and unselected distinct algorithms stay unresolved.

All first-invocation sample phases remain **unknown**. Source closure inclusion is
not proof that the measured invocation was in a retained trial. No full elapsed
ROI, complete launch census, causal resource limit, measured intensity--speedup
correlation, source/numerical correction or new numerical acceptance is claimed.
Source integer/atomic/SFU work is not silently reclassified as scalar FP work.
