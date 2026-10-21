# Q–Z source-family membership proposal — private, root merge only

`membership_patch.json` supplies **31 timed-family stage updates across23 model
records and six names**: quant3MatMul, quantBnB, quantAQLM, randomAccess, svd3x3,
wmma. quant3MatMul/CUDA.jl is explicitly held: its selected profiler parent failed,
so no source membership is promoted for that slot despite available counter fragments.
No source/ROI/failure/correctness status is changed by this private packet.

## Merge prerequisites and shape

- `stage_updates` is keyed by exact existing benchmark/model/stage_id. Each has
  parent gzip review **compressed+raw SHA**, parent canonical record SHA, old stage
  SHA, and minimal proposed fields. Require the old hashes at merge; do not overwrite
  concurrent reviews. Root must take the metadata lock and check selected capture,
  config/result, source and parent-review consistency. Hold on drift.
- Every promoted stage is source_timed_closure_inclusion_proved with explicit
  ROI_membership_proof and semantic_family_in_primary_ROI:true. Model function names
  are exact regex tokens against the saved demangled NCU identity, not guessed
  blanket substring rules. Corrected regexes have single regexp escapes.
- `duplicate_regex_alias_updates` separately proposes ten overlapping callsite
  candidates become model_kernel_regex:null and family_alias_of canonicalstage.
  Their source metadata remains intact; this does not promote their membership or
  label an actual warmup as timed. For example native `VecQuant3MatMulKernelFaster`
  is a DIFFERENT function and is **not** treated as an alias. The unused faster
  path stays unresolved. Parent may review/apply aliases separately, but canonical
  mapping must not acquire multiple matching families.
- `evidence` has exact selected ledger/profile/manifest/plan/provenance/derived
  SHA bindings, raw config/result SHA, actual target/build argv and source excerpts.
  Source bytes were checked against plan or captured additional-before-profile
  bindings, then read at their unchanged current bytes. This is not new current-
  source backfill for an old profile without captured identity. Frozen compressed/
  raw member SHA and lengths were verified. No parser, target, JIT or GPU was run.
- The initial metadata shared-lock query waited behind an exclusive holder, then
  finished without interference. The actual proposal generator reads immutable
  parent members with selected ledger before/after SHA stability. Label this
  **sequential exact-byte proposal**, not a simultaneous coordinator snapshot.

## Source-specific conclusions

1. quant3MatMul: factory creates operation!; timed launch! zeroes output AND invokes
   the quant3 kernel. OwnedStudy repeats that closure within its host timer and
   synchronizes. Native adapter does the same through a launch lambda. Thus the
   quant3 family is included, but reset traffic/work is also timed. No whole-ROI
   positioning follows from quant3 counters alone. Faster native variant unmeasured.
2. quantBnB: selected factory returns the quantization kernel; OwnedStudy's timed
   repeated launch closure includes it. Codebook parsing, CPU reference and readback
   conversion are outside. Native adapter brackets its same launch lambda.
3. quantAQLM: both case/four-vector dispatch paths pass launch into the source timer,
   with device completion before elapsed accumulation. GPU input echoes/guards and
   CPU quality audit are outside. Existing corrected-reference/data-stride scope
   remains; the patch does not make it an untouched native case.
4. randomAccess: run! launches init then update; the timer brackets run! itself.
   Consequently **init is a timed family**, not an excluded one-time setup kernel.
   CUDA.zeros' runtime fill is a different setup family and remains unassigned by
   these existing-stage patches. This proof does not assign all fill kernels or
   infer raw update multiplicity/steady-state phase.
5. svd3x3: selected launch_svd maps each model to factorization family. run_svd/work.run
   repeats only that dispatch inside its clock and device-completion boundary;
   scientific property checks, resets and artifact writes are outside. Source warmup
   calls the same family before trials. Nonunique raw factorization is not recast
   as identical raw numerical outputs; original property contract stays intact.
6. wmma: case-dependent impl0/impl1 factory closures are called inside each selected
   case timer. Preparatory30launches and native off-contract diagnostic can use
   the same function family outside the timer. Both primary families are members,
   but first sampled invocation phase remains unknown, including possible diagnostic.
   This patch does not enable tensor-FLOP conversion, aggregate cases/repeats or
   claim source matrix bounds are valid beyond the selected contract.

## Nonclaims

All first-invocation sample phases remain **unknown/not proved inside a primary
retained timed invocation**. Membership of a function family in the primary timed
closure does not prove the sampled call happened there. No warmup exclusion is
inferred for a family also used during timing. No new whole-ROI, steady-state,
complete launch census, causal bottleneck, equal-weight kernel mean, performance
speedup correlation, hardware/framework availability or numerical acceptance claim.

Saved identity matching is a CPU evidence annotation, not a new benchmark test or
profiling run. All proof/status changes are proposal-only. No shared tool, review,
ledger, paper, source or Git edits; GPU/build/test count0. Prospective shared
annotation usage is separately closed and retained in usage_receipt.json.
