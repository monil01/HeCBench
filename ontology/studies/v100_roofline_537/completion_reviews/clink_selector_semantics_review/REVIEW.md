# clink KA selector semantics: qualified interpretation

## Findings

1. Installed **2022.4** primary CLI documentation defines invocation-nr per kernel function, with per-launch-config applying filters independently by grid, block and shared-memory configuration. The mangled-name option selects the name basis used for matching; it does not itself enumerate runtime functions. The `.*` regex matches any name, and final `1` requests its first matching invocation under the filter policy. The empty context/stream fields do not establish per-stream or retained-trial selection. Therefore the existing label `firstper_mangledspecialization_andgeometry` has a reasonable **documented sampling-intent** interpretation, but is too strong if treated as a proved complete inventory of every authored or generated specialization.
2. Installed **2025.2** wording is more explicit about kernel-name and launch dimensions/shared bytes entering invocation counting, excluding context/stream IDs when omitted. It does not document grouping all distinct kernel names together merely because geometry is the same. This new wording cannot retroactively certify the old implementation. `per-launch-config` includes shared-memory configuration, not only grid dimensions; the old short label omits that qualifier.
3. Existing saved selector probe provides a limited counterexample to blanket geometry-only aggregation: `per_config_tmp2.csv` retains both `selector_fp32` and `selector_fp64` at grid256/block256, and `shape_combined.csv` retains both functions at each of grid128 and256. Its source launches both functions repeatedly. These old records were only read, not rerun. Important limitations: saved probe commands omit explicit `--kernel-name-base mangled`, and small provenance does not bind exact executable version/binary/compiler identities. Treat this as retained empirical evidence, not a new exact-method or Julia-family calibration.
4. Exact clink KA batch003 command has wildcard first invocation, per-launch-config and mangled-name matching; status is completed, rc0/no stop. The raw wide CSV has one actual numeric-ID kernel row, plus its units row. It is not a row-loss artifact. Its lone signature has one array and two Float32 scalars, while the current source-bound authored KA kernel accepts three arrays; this remains **unresolved**. No docs-only argument proves that it is fill, initialization, LSTM arithmetic, source lowering, compiler argument elimination or selector collision. No SASS or new runtime observation was obtained. All five full application checks may pass while the profiled sample is outside the timed mathematical callback. Preserve root's unknown-family/ROI hold and all original metrics/commands/checks.

## Proposed descriptive qualification (no historical rewrite)

Keep each actual historical argv unchanged. Supplement the method description with:

> First matching invocation under the recorded wildcard kernel-ID and per-launch-configuration filters, using mangled-name matching. The saved launch set is a partial sample, not proof that every kernel specialization, timed callback, stream or geometry was captured.

If retaining a compact label, qualify `firstper_mangledspecialization_andgeometry` as intended selection policy only and explicitly include block/shared-memory configuration. Do **not** replace it with an assertion that wildcard invocations aggregate all families by geometry: neither inspected primary document nor the saved probe proves that. clink KA should remain excluded from family-qualified timed-ROI interpretations until source-to-observation identity is actually demonstrated. No mathematical/profile acceptance change or automatic rerun is recommended by this review.

## Primary source locations and provenance

- Installed 2022 official CLI: `/usr/share/doc/nsight-compute/NsightComputeCli/index.html`, lines975–1044 and1075–1095.
- Installed 2025 official CLI: `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Compute/docs/NsightComputeCli/index.html`, lines936–968 and983–1005.
- Public corresponding NVIDIA documentation routes: `https://docs.nvidia.com/nsight-compute/2022.4/NsightComputeCli/index.html` and `https://docs.nvidia.com/nsight-compute/2025.2/NsightComputeCli/index.html`. This review read the installed copies; the links were not fetched and no web-version byte equivalence is claimed.
- Exact line excerpts and whole-file SHA/lengths: `installed_primary_document_excerpts.json` and `review.json`; all source/capture/probe bytes stable before/after review. Source excerpts are preserved for internal root review, not proposed verbatim manuscript text.

No GPU query/run, build, benchmark test, shared edit or acceptance change occurred. Usage is shared profiling-method review, separate from translation.
