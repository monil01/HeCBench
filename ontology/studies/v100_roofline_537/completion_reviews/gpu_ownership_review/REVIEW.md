# ASSERT045 GPU ownership stop — evidence and prospective instrumentation

**CPU-only private review.** No candidate/probe imported or executed; no reruns,
GPU queries, builds/tests, shared tools/helpers/ledgers/source/Git edits. Root's
queued calibration still binds collector32a9... and is untouched.

## Historical finding: ownership unresolved, not a proved false positive

Exact immutable hashes/logs/receipts are in `historical_evidence.json`.

- ASSERT045 CUDA.jl prephase ran from19:42:08.601930 to19:42:36.201612UTC.
- Three completed negative-process records each have actual returncode0 and exactly
  ASSERT_STATUS710 with complete raw streams. Third finished19:42:35.628661.
- Guard recorded NVML PID**1286308**, process-name**[No data]**, at19:42:35.671311,
  **42.650ms after the third completion receipt**. This is ordering evidence, not
  proof it was that third process or that a foreign process did/didnot execute.
- Completed subprocess receipts1–3 have **no hostPID/startticks/SID/PPID**. Third
  stderr contains CUDA block/thread assertion coordinates, not a hostPID. Do not
  confuse GPU thread coordinates with Linux process identities.
- There IS partial fourth-child evidence: `negative_raw/4.stderr`979B starts
  `[1286419] signal15:Terminated`, during Julia system-image initialization with
  Allocations0. This self-reported hostPID is **different** from NVML1286308.
  It lacks a birth/start/SID/PPID receipt; it does not identify completedchildren1–3.
  Fourth stdout is0B. No completed fourth/fifth check or successful full prephase
  exists, and the positive NCU phase was not launched.
- Original prephase returncode-15/stop_reason
  external_gpu_activity_detected_own_profile_stopped and whole collection failure
  remain unchanged. There is no valid basis to promote045 or claim a known guard
  false positive. The [No data] string is not an ownership proof in either direction.

The precise **old ancestry-check failure reason was not retained**. The original
code silently catches OSError/ValueError; it could not resolve SID, /proc ancestry
or could have observed a genuinely different live process. Never backfill a
ProcessLookupError, exited PID or specific root cause as if observed.

## Source-level race that can recur (possibility, not retrospective diagnosis)

Collector.run samples NVML, then separately calls getsid(PID) and reads current
/proc ancestry. Phase helper uses subprocess.run, which waits/reaps each direct
negative child before retaining its output receipt and spawning the next child.
The NVML observation and later /proc lookup are not atomic. A child may exit/reap
between them, leaving no SID/ancestry for the later lookup; original code then
sets owned=false. This is a source-derived possible execution order, **not proof
it happened for PID1286308**. No claim about NVML's guaranteed staleness semantics
or a numeric race probability is made. No extra sleeps/name filters can prove an
unknown PID was ours.

## Minimal prospective candidate packet

1. `phase_helper_telemetry_candidate.py` + `phase_helper.diff`:
   copy of exact9ad224... helper. Negative child invocation becomes Popen plus
   explicit wait, preserving argv/env/stdout/stderr/isolated contexts and all five
   original710+exit checks. Persist parent identity and active-process attempt
   **before spawn**, then actual Popen PID and /proc identity **before wait/reap**;
   retain after-reap identity or exact read error. Birth record includes bootID,
   PID/startticks/SID/PGRP/PPID/UID/state and command SHA when readable. Exited-but-
   unreaped children can still produce a Z-state birth record; missing data is
   explicit, never fabricated. Partial active child receipt survives phase stop.
2. `ownership_probe.py`: read-only process snapshots, before/after stat consistency,
   root leader birth/current identity comparison, actual ancestry chain/error
   snapshots and PID-start change detection. No GPU queries, signals, allowlist or
   names-based ownership decisions. PID/start/boot joins are evidence, not magic
   recovery of missing historical generations.
3. `collector_instrumentation_candidate.py` + `collector.diff`: private copy of
   exact32a9cf... collector. Retains leader identity, every NVML-query interval and
   parsed row, original guard decision, current identity snapshots and ancestry
   errors, including disappearance/changed generation. Proposed acceptance is
   **legacy_owned AND stable_current_owned_identity**: a strict intersection,
   never a relaxation. Unknown or observed PID reuse still stops the attempt.
   Root stop_reason remains unchanged in this minimal candidate; additional
   classification distinguishes unresolved evidence without claiming unique cause.

These candidates make the next failure auditable; **they do not by themselves
eliminate the exit/reap race**. A child can still disappear before observation;
that stays unknown/stopped, even when its earlier birth receipt suggests a plausible
join. No lookup is accepted solely from a prior PID ledger. Source syntax/flow was
inspected only; candidates have not undergone runtime validation. Do not install
these copies over currently pinned sources or mutate old037/045 plans/reviews.
A root adoption requires fresh exact helper/collector/phase bindings and new receipts.

## Optional safer lifecycle design — proposal only, not implemented

The smallest way to avoid losing an owned negative child's /proc identity is to
retain that **specific direct child unreaped** until the collector has observed
completion/clearance, rather than accepting a historical PID after it disappears.

1. Keep a strong Popen object for the exact child. Capture its birthPID/start/boot,
   parent/session links immediately, optionally obtain a pidfd. Preserve any unknown
   as unknown; do not start new GPU work until the collector validates the birth
   identity if using a startup gate. A pidfd targets a kernel process instance but
   is **not** a policy to accept a reused numeric PID or a stale NVML row.
2. Observe termination with Linux waitid WEXITED|WNOWAIT (prefer a pidfd-associated
   wait where available), **not Popen.wait/poll or another reaping thread**. The
   direct child remains a zombie, retaining /proc identity and reserving its numeric
   PID. Record exit status, start/SID/PPID and stateZ; do not run that context again.
3. Parent helper publishes a nonce/attempt-bound exit-awaiting-reap record. Collector
   under the same existing GPU lock revalidates exact root/child process instances,
   UID/SID/parent relation and own-session ownership on its ordinary GPU query.
   A current unreaped zombie's SID remains inspectable. Foreign/unknown/reused
   rows continue to fail closed; no [No data], same-user or “recent PID” exception.
4. Only after a **post-exit** query contains no record for the held child, all other
   GPU rows satisfy the unchanged guard, and source/attempt identity still matches,
   collector writes an exact birth-identity acknowledgement. Helper then reaps that
   child, checks real710/exit0 outputs and starts the next isolated child. Keep
   both raw queries/ack and child's actual exit receipt. No query/name equality
   substitutes for the original scientific assertions.
5. If the process identity disappears, another thread reaps early, handshake data
   is missing/mismatched, a row reappears after reaping, deadline/reserve/foreign
   guard trips, or cleanup is uncertain, stop/preserve failure. No timeout-based
   assumption of ownership and no continuation from an unresolved counter window.
   Root's own-session termination/quiescence policy stays in force; no foreign
   process is ever signaled. Interrupted negative contexts never become passes.

This is a more involved **host lifecycle instrumentation/method revision**, not
an arithmetic change. It needs source review and fresh method validation before
adoption. It is not delivered as a secretly active patch or a proved fix. It should
not be mixed into pinned32a9 calibration/imports mid-queue. Unknown NVML PID epochs
cannot be made certain by a cache entry, delay or process-name heuristic.

## Reporting recommendation

Describe this archived case as an **ownership-guard stop with unresolved process
identity**, retaining the literal original stop_reason alongside that caveat.
Do not count it as a numerical/compiler/framework failure or proof of actual
outside GPU interference. Existing measurement census raw stop categories remain
unchanged; any later paper/derived taxonomy refinement belongs to root after current
hash-bound exporters finish. Shared review effort is separate from translation,
compiler/JIT attempts, calibration runtime and retrospective per-model allocation.

## Primary Linux lifecycle references

The proposed unreaped-child design relies on documented Linux wait semantics, not
an observed validation of this candidate. `waitid(..., WNOWAIT)` leaves the child
waitable for a later reap; see [waitpid(2)](https://man7.org/linux/man-pages/man2/waitpid.2.html).
The optional post-exit pidfd acquisition additionally requires that SIGCHLD is not
explicitly ignored, SA_NOCLDWAIT is unset, and no other handler/thread has already
reaped the child. Those conditions must be checked before adopting the design; see
[pidfd_open(2)](https://man7.org/linux/man-pages/man2/pidfd_open.2.html).
A pidfd does not establish ownership of an unrelated numeric PID returned by NVML.
No waitid/pidfd lifecycle experiment was executed in this review.
