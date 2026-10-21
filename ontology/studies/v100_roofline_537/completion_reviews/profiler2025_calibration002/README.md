# Fresh private calibration002 controller proposal — not launched/adopted

`delivery.json` contains the exact ROOT-ONLY future argv, controller/manifest hashes and fresh raw output path `/tmp/hecbench-roofline-v100-20261020/method_calibration_ncu2025_2_0_002`. That output directory has not been created by this worker. No controller import, GPU/query, build, test, shared tool/source/ledger edit or Git operation occurred. Only text-authoring and CPU file inventory scripts were executed.

## Actual001 failure / preservation

Root's completed001 attempt failed PRETARGET with `RuntimeError('GPU6 identity differs')`. Its `precision/identity.log` actually contains the correct GPU6 UUID/PCI row, preceded by one source-defined Landlock guard log line. The old parser treated that prefix as an extra CSV row. This is an observer/identity parsing failure, not a GPU mathematics failure or evidence of wrong hardware. No counter or full-host canary pass is inferred from the identity line.

Every original001 file and prior proposal remains untouched. `preserved001/` holds lossless private copies; `preserved001_inventory.json` hashes the originals and copies and records exact source identity. `controller.diff` shows only the new normalizer and its identity/version uses. The original controller is preserved with SHA `eaf532a0ac9ef114045fc7e12842191edcd667a29821eecace88f387d646a630`.

## Minimal002 change

New `guarded_payload()` accepts exactly one FIRST prefix line beginning `V100_WRITE_GUARD ` and validates its actual JSON keys:

- `landlock_abi`: integer >=3;
- `write_directories`: exact vector from this controller's guard (`[actual fresh002 output directory]`);
- `privilege`: exact `existing unprivileged user; no_new_privs`.

There is no fabricated `policy` key and no arbitrary warning/row skipping. Missing, repeated, malformed or changed prefix fails closed. Full original merged log bytes are preserved. Each receipt records raw log SHA/size, exact prefix SHA, normalized payload SHA/size/file. The normalized payload is saved separately as `identity.payload.bin` or `version.payload.bin`.

Identity parsing then requires exactly one UTF-8 CSV payload row and the same exact GPU6 UUID; original PCI interpretation is unchanged. Version parsing requires exactly three known-format payload lines, exact NVIDIA header, copyright format and exact version2025.2.0.0 plus numeric build/public-release format. Full merged version log remains `actual_profiler_version_text`; separate `actual_profiler_version_payload_text` and SHA identify the normalized version bytes. This is recovered payload from a merged log, not a false claim that stdout/stderr had been independently captured.

## Unchanged scientific/execution scope

The contiguous `shape`/kernel-key/metric/scientific validation function source is verified byte-identical. All nine scientific calibration rows, all full collector metrics, old2022 invariant comparisons, expected exact counts, packed-half/tensor factors, canary binaries/source, original complete host-check lines, pinned PATH/LD environment, UUID/PCI checks, globalGPU6 lock/foreign-process monitoring, Landlock policy, fresh output rules, process deadlines and2GiB reserve remain unchanged. No metric dropping, source rebuild, kernel change, acceptance relaxation or guessed compiler identity.

The calibration has the existing two complete host-check processes and7precision+2half-arithmetic identities/geometries; it is not a new five-trial benchmark timing series. Primary benchmark five-run timing data remain untouched. Original compiler identity for existing canary binaries remains unknown, explicitly retained.

Manifest preparation binds83 files again. Only the controller/prospective preparation paths and associated hashes change; collector stays32a9, SDK/canary/reference values stay the same. Root's ongoing summarize/audit work is not a canary-bound input and is not modified here.

## Root next step

Review exact source/diff/manifest first; launch only through `delivery.json` argv with unchanged safeguards and currentGPU6 authorization. Do not disturb foreign GPU work. Any new failure is retained as002; do not overwrite001 or claim pass/adoption. Passing source-normalization review alone is not scientific calibration success.

The private method-patch v2 still hard-pins the001 controller/manifest. It intentionally cannot admit002 until root separately reviews an actual successful002 calibration and rebinds its gate/archive controller and manifest identities. No method adoption or shared patch application is made here.
