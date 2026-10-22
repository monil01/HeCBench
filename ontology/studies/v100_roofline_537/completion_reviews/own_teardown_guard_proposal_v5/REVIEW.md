# Private ownership guard v5: monitored scanner only

Status: SOURCE PROPOSAL, root and independent review required. No staging, candidate import, execution, GPU query, signal, test, build, or canonical edits occurred.

This is a seven-literal-edit successor to frozen v4. It addresses the independent v4 HOLD: baseline own_session_members swallowed per-process errors, and refresh silently discarded unpinned candidates. The original baseline scanner text remains byte-identical and is used when monitor_gpu=False. No arithmetic, kernel, oracle, timer, metric, selector, profiler, UUID, or scratch-manifest policy changes are proposed.

## New evidence handling

The monitored-only scanner enumerates /proc and returns separate live session members, unresolved candidates, and confirmed process-directory disappearances. A missing stat or namespace file is not directory disappearance. Permission errors, parse errors, changed membership snapshots, and failed directory-absence confirmation remain unresolved. Known other sessions or UIDs are excluded before full identity reads for untracked processes; there is no claim to manage all system processes. A candidate that cannot be excluded is conservatively relevant. Previously tracked relevant candidates cannot be excluded by a later changed SID/UID; that remains unresolved and cannot transfer an ownership token.

Refresh records exact scan diagnostics, retains unresolved numeric candidates only for quiescence, and raises instead of silently continuing. This state is not ownership proof. Signals retain v4 live generation + pidfd + fresh-session checks. An unresolved scan supplies rejected entries and delivers no signal from that scan. Quiescence uses the strict scanner only for monitored runs. The unbounded GPU-lock safety hold also uses it, including previously unresolved PIDs and pinned records. No numeric PID, missing /proc file, or dead pidfd alone admits an NVML row or authorizes a signal. Confirmed directory disappearance allows a safety hold to clear, never converts a failed attempt into admitted counters.

## Source proof and boundaries

source_inverse_proof.json records all seven exact before/after replacements. Reversing them restores every v4 byte; composing the frozen v4 inverse restores every canonical C21 byte. The original own_session_members function is identical. Source-preparation code only read/wrote source text and JSON and computed hashes; it never compiled, imported, or ran the candidate.

V4's fail-closed treatment of ambiguous NVML PID generations, pidfd-only monitored signaling, bounded active quiescence followed by an explicit unbounded lock hold, no unknown signals, and no freeze/cleanup while unknown writers remain are retained. A stopped/crashed supervisor cannot retain a flock; operator reconciliation of a hold marker remains necessary. PID reuse can conservatively prolong a hold. /proc enumeration is a sequence of snapshots, not an atomic proof that no process can fork; this patch makes observed uncertainty explicit and does not establish race completeness. Unprivileged /proc access errors can cause conservative holds.

## Scientific admission remains prospective

Historical batch064 remains failed. Its five correct rows and numeric PID correlation do not identify an NVML row's process generation. This source change has no runtime calibration and does not guarantee TGVNN completion or demonstrate foreign-process/PID-reuse/race behavior. Frozen075 and planned076 remain separate, held future authorities. A small future accepted triad native run must bind exact current method/source/selectors and collect all five full checks plus live ownership/bootstrap/teardown/quiescence evidence before root considers any deployment. No canonical C21 replacement is proposed by this packet.
