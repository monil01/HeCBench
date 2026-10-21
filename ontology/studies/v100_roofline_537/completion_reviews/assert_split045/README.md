# Fresh ASSERT045 — path-only relocation, no execution

Fresh root: `/tmp/hecbench-roofline-v100-20261020/batch045-assert-isolated-phases`.
Exact plans/review/phase hashes and retained037 failure receipts: `delivery.json`.

All source CUDA/Julia kernels, common assertion controller, original Python checker,
config/result hashes, exact installed Julia ELF, pinned project/preferences,
phase helper SHA and authorization marker remain unchanged. The helper is reused
at its exact immutable037 path, **not copied/patched**. Only fresh scratch/output,
phase-spec/receipt/plan/review paths change. Original037 paths and failure evidence
are not overwritten. `plan_diff.json` provides full old/new plan text difference.
All declared bindings were rehashed during CPU-only staging; source bindings and
positive argv equality were checked exactly. No target/helper phase/build/test ran.

Parent observed037CUDAjl fail before any negative child: relative guard executable
`tools/v100_profile_write_guard.py` was resolved under scratch and could not open.
This is parent collector path construction, not a Julia/device compiler rejection,
unsupported framework, actual failed710 assertion or numerical mismatch. Parent
owns the separate collector path-normalization fix. No shared tool was edited here.

## Root launch after its collector fix

```
python3 /home/imo/HeCBench_julia/tools/v100_roofline_collect.py \
  --source-root /home/imo/HeCBench_julia \
  --plan-root /tmp/hecbench-roofline-v100-20261020/batch045-assert-isolated-phases \
  --isolated-assertion-reviewed --timeout 600
```

Use **absolute source root**; root alone launches. Do not wrap collector in another
GPU lock. Its existing lock, identity/foreign-activity monitoring, fresh write guard,
source before/after validation and mandatory pre/post hooks remain required.

All five real negative fresh contexts (exact710 and exit0, complete raw receipts)
must pass before NCU. NCU runs original positive Julia entry, producing all five
actual PERF_TRIAL rows. CPU postjoin rechecks negative bytes/status and positive
rows, then emits original execution-only contract. Missing/failed/partial evidence
never becomes accepted numerical data. Persist negative_raw and slot-top phase
logs/JSONs on failure too. Retain original accepted performance series unchanged;
instrumented positive timings do not replace five-run means or establish complete
ROI/causal coverage. Native ASSERT prior selected profile remains untouched.

The only opt-in gate is `requires_isolated_assertion_collector_phase_hooks`.
Source review does not assert runtime success; root must preserve actual new
statuses and evidence. Shared staging usage is separate from translation/compiler
attempts and recorded in usage_receipt.json after its own prospective window closes.
