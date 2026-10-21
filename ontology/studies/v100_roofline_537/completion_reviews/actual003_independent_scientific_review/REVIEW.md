# Independent scientific review of saved actual calibration 003

Status: saved evidence consistent; no scientific blocker identified for root's separate admission decision. This is private review evidence, not an approval receipt or live method adoption.

## Evidence examined

The independent standard-library reconciler read frozen actual003 controller, collector and parser, raw execution/version/identity logs and payloads, both wide counter CSVs, prior 2022 counter tables, exact canary sources, 83-entry source manifest and v5 proposal as source text. It did not execute or import the v5 helper, packer, controller or canaries. No GPU process, build or target test was launched.

The evidence has nine distinct source-family/launch-geometry rows: seven precision and two packed-half arithmetic. Each supplies all 27 requested metrics with finite nonnegative values, correct units and positive duration. The full raw host check lines match the source's complete-array validation. All nine source-derived typed instruction counts and the 48 selected historical counter invariants agree exactly. DRAM read plus write equals direct sum on all nine saved rows.

Actual version payload is Nsight Compute 2025.2.0.0 build 35613519, public release. The device UUID, PCI address and sm70 capability match the v5 constants. All 83 historical before/after bindings match; current files also matched during this independent audit ending 2026-10-21T23:19:44Z. That last result describes the audit instant only: deployment intentionally changes live collector/helper/summarizer/audit. Historical calibration remains bound to its frozen collector 32a9cfeb..., not the later v5 collector c21b1edd.... Frozen input archives and v5 manifest/controller constants join exactly.

## Scientific interpretation

- Packed HFMA counts 524288 instructions and 2097152 physical scalar operations in both half2 and scalar-half cases. Scalar-half useful source work is 1048576, so the source distinction must not change the physical packed factor.
- The tensor row has 2048 counted tensor instructions and 1048576 scalar operations. The factor 512 is justified only for the exact Volta HMMA884 context and source/geometry checked here.
- Packed hadd2 and hmul2 each have 524288 instructions and 1048576 scalar operations. Float and double selectors each have 4096 FMA instructions and 8192 operations for both checked geometries.
- Three frequency metrics use raw unit `hz` on each row. Exactly 27 factor-one unit views to `cycle/second` are justified inside this method; raw CSV bytes and values remain authoritative.

## Operational evidence and limits

Recorded guarded version/identity/profile subprocesses exited zero with no stop reason and no recorded external GPU activity. The exact Landlock ABI 4/no_new_privs prefix is retained alongside normalized payloads; no arbitrary warning removal was used. Fixed environment, private output directories, minimum 2048 MiB free-space gate and operational deadlines are supported by frozen source and receipts. The 4236-second common GPU lock wait is outside case durations (3.804 and 2.090 seconds). This is a saved-source/receipt review, not independent replay of historical process ancestry.

Raw CSV warnings remain visible: stock section deployment to Documents failed, installed SDK sections were used, and GPU clocks were unmodified. Bound fallback sources and complete requested metrics were checked. Timings, clocks and traffic are not equality invariants between profiler versions; this evidence supplies no sustained performance ceiling or timing equivalence. Original compiler executable/version is unknown because existing binaries were reused. Scope is one V100 sm70 and the exact existing canaries/profiler build; no full-ROI, causal or universal method claim follows.

Root owns canonical packaging, its approval receipt, deployment and post-deployment validation. This review does not attest those later actions. Detailed immutable input hashes, process joins, row values and v5 source hashes are in review.json and science_rows.json.
