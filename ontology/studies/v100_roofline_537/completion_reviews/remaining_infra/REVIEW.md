# Remaining infrastructure: EGS and ASSERT

CPU-only retained evidence/source review and preparation of seven fresh plans. **No GPU, build, benchmark execution, tests, source-contract changes, accepted-source edits, shared tool/ledger edits or Git.** Existing bound helpers/plans were not modified. New independent scripts are only in this private review folder.

Staged plan root: `/tmp/hecbench-roofline-v100-20261020/batch030-egs-assert-infra`.
Exact parent/new plan, config/result, log, counter, source patch/diff, helper and manifest hashes: `delivery.json` and each `remaining_infra_source_proof.json`.

## Findings and taxonomy

| Case | Actual observation | Not established | Prepared intervention |
|---|---|---|---|
| EGSnative | Unconditional independent C++ checker compiler child exits255 before source warmup; no explaining compiler stderr | Source math/compiler bug, temporary-path denial, profiler-injection cause, feature unsupported | Compile the exact checker and flags before NCU under existing write guard; require its build/source hash before target |
| EGSJulia3 | Same compiler exit255 in finalizer after first completed captured trial's full-physics check | Device/scientific kernel failure, precise compiler termination cause | Same prebuild; existing finalizer already skips compile when mtcheck exists; retain every independent MT/physics audit |
| ASSERTJulia3 | First isolated negative child produced empty stdout/stderr, with no ASSERT_STATUS; old wrapper did not persist signed returncode; positive branch not reached | Actual code710 changed, assertion launch/replay reached, unsupported Julia/CUDA/NCU, exact launcher or injection cause | Full original checker retained, direct exact installedJulia1.12.6, and complete raw perchild diagnostics written before checker rejection |

All seven plans are **unexecuted, unproven infrastructure retries**. No original failure is relabeled successful. No root-cause attribution is inferred from the proposed controls. A later successful result would establish recoverability under the revised profiling setup, not isolate the original causal mechanism without additional controls.

## EGS: unchanged independent checker, outside target-process instrumentation

The original compiler command is:

```
g++ -O3 -std=c++17 /home/imo/HeCBench_julia/src/egs-julia/check_mt_trajectory.cpp -o mtcheck
```

The checker source is an independent serial UInt32 MTGP recurrence. It reads actual captured initial state/parameters/tables, checks all101live words and27unchanged trailing words for each actual phase, and must find the state within the source-declared finite search. It includes no CUDA headers/kernel body and performs no GPU calls. Its checks and search bounds are unchanged. This is recurrence membership, not exact random-draw count, distributional quality, or a full independent transport trajectory; those limits remain.

Each fresh EGS build command invokes `build_egs_checked.py` under the existing Landlock guard. A SHA-bound `egs_build_manifest.json` contains:

1. The original selected host-preparation CUDA shared-library compiler command and flags. Executable `nvcc` is canonically resolved to existing `/usr/bin/nvcc`; no compiler/toolkit stratum or numeric flag change.
2. The exact original CPUchecker source and `-O3 -std=c++17`, writing only fresh scratch/mtcheck.

The launcher records each actual build returncode, output SHA, source SHA and command in `extra_build_outcome.json` when the root eventually builds. It refuses existing outputs and validates bound inputs. No compilation has occurred during this review. No private child environment override or loader knob was invented.

Before any target begins, `checker_gate.py` requires the compiled checker and library hashes to match that actual build record and verifies their input hashes, then execs the original target command. The native Python copy preserves canonical readonly imports/HERE and replaces only its redundant in-profile compile with an explicit requirement that the checked prebuilt mtcheck exists. Julia finalizer source is unchanged: its existing conditional skips compilation because mtcheck already exists. Both still execute `./mtcheck` and all other independent physics, full-field, inputs/readonly/guard checks in each of the original five trials.

All source GPU kernels, actual input readbacks, captured fields, RNG checks, tolerances, iteration/phase loops and declared ROI remain intact. The CPU checker was already outside measured simulation/accumulation segments; moving its compilation earlier does not remove any scientific computation from that ROI. The compile is now before CUDA initialization/profile target-tree interception; this joint ordering/interception change is disclosed and does not identify which caused exit255.

## ASSERT: keep the checker; improve observability and remove dispatcher ambiguity

The new copied wrapper retains the complete original checker remainder **exactly**, except replacing its two subprocess.run call sites by a capture wrapper. The static packet audit confirmed all three copies preserve that full suffix:

1. Run five separate `--negative` processes.
2. Each must exit0 and emit exactly `[710]` from actual ASSERT_STATUS lines; otherwise reject.
3. Run a separate positive process.
4. Require its returncode0 and exactly five PERF_TRIAL rows.
5. Emit the original execution-only attestation/negative-diagnostic fields and original measured times, with no invented numerical arrays or errors.

All device sources and `study_common.jl` are unchanged. Safe testN64 precedes intentionally failingN60 in the negative process with the same32threads×2blocks and dynamicInt32argument. The unchanged first-invocation-per-function/geometry profiler selector is intended to select only that safe first invocation, not replay the failing second launch. This is a source/selector expectation, **not an observed guarantee** across backend specializations/contexts. NCU may still reject the intentional CUDA error; if so retain the new failed attempt. No CUDA assertion error is suppressed or reclassified to pass.

The Julia command changes only from version-manager `julia` dispatch to the exact installed executable `/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia`, which is SHA-bound. Project, source entry, arguments, assertion status, geometry and check ordering remain unchanged. This removes a dispatcher layer but does not prove the original failure was caused by it.

`_captured_run` captures **raw stdout/stderr bytes**, signed OS returncode, command and timestamps before returning to the old checker. It writes them as full base64 plus lengths/hashes to a slot-side `assert_child_diagnostics.json` **outside disposable scratch**. Thus even an empty-log startup failure will have the actual signed status. Spawn exceptions are also persisted before rethrow. Original combined text logs still exist in ordinary profile_outputs during execution. This newly observed corroboration would not reconstruct the old missing raw child logs.

Existing native ASSERT profiling succeeded and was not duplicated. No positive-only bypass was staged. All three new Julia attempts remain subject to their full negative and positive checker; full correct execution can still fail as a profiler method, and counter/ROI absence is not synthesized into zero metrics. Assertion-only source has no numerical outputs; conventional floating roofline may be inapplicable, and neither this source review nor a successful partial counter implies all12positive warmup/timed launches or dynamic assertion count.

## Parent adoption

Use the seven already staged plans after inspecting their exact diffs and manifests. Parent owns all builds/GPU launches and acceptance/ledgers. To regenerate into a fresh tree (never overwrite the staged one):

```
python3 /tmp/roofline-completion-20261021/remaining_infra/prepare_remaining.py /tmp/NEW-FRESH-REMAINING-PLANS
```

The planner executes no subprocess, compiler, GPU API or target. The compiler-only launcher is present for **future root builds**, not called by the planner.

Preserve standalone helper/template sources, each `egs_build_manifest.json`, `remaining_infra_source_proof.json`, `plan_diff.json`, future `extra_build_outcome.json` and especially future `assert_child_diagnostics.json` when freezing. The latter contains full stdout/stderr bytes and should not be reduced to hashes alone. All plans are currently frozen; do not mutate their bound scripts while queued.
