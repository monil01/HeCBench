# Retained trial-2 scientific-family sums

Final reviewed snapshot: `reviewed/audit.json`, `reviewed/summary.csv` and
`reviewed/PROTOCOL.md`. The default CLI uses the live metadata shared lock,
rehashes every bound input before releasing it, then writes a fresh `/tmp` tree.
`INTERFACE_SCHEMA.json` specifies the CLI, admission fields and aggregate fields.
No existing output is overwritten; root can export fresh then atomically copy.

Current admission: **three** complete source-counted scientific families, all
Triad (CUDA, CUDA.jl and JACC), 100 launches each. Each contains 838860800 executed
FP32 FLOPs, which exactly matches the separate useful-work expression 2*N*100.
Actual cold replay DRAM totals are respectively 4873407776, 4873709888 and
4873876704 bytes; integrated intensities are 0.17213023, 0.17211956 and 0.17211367
FLOP/byte. Sum of replay kernel durations: 0.006499424, 0.006478624 and
0.006450368 seconds. These are counter-defined totals, not averaged kernel
intensities, historical host-clock performance or causal bottleneck labels.

KA retained Triad failed, so its old partial capture is unaggregated. BabelStream
and Jacobi selected captures are still partial; no one-kernel multiplication is
allowed. BabelStream is dot-only and fits L2, not the full STREAM suite. A future
complete Jacobi stencil window still excludes timer-included resets, scalar D2H,
host norms and non-conventional floating-atomic costs. No full elapsed-host ROI
is claimed for any case.

Original five-run means, fresh instrumented trial2 timer intervals and summed
replay kernel durations are distinct fields. The fresh profiler rows all report
correct CPU selfchecks, but disposable outputs were removed after manifesting;
this audit does not invent a fresh direct-artifact dual-reference certificate.

`input_hashes` binds all original config/result/source, frozen compressed and
uncompressed profiler data, recipe and selector inputs. Each launch is summed
once (NCU replay passes do not multiply scientific launch counts). Profiles use
cache-control all / clock-control none; physical traffic is replay-conditioned.

Earlier `snapshot/` is a preserved exporter-development output rejecting archived
wide CSV before wide-format support was added; `final/` precedes the extra exact
executed-selector/timing-normalization checks. These are CPU exporter revisions,
not GPU candidates, numerical failures, or additional experimental measurements.
Only `reviewed/` is the delivered audit. No source, GPU, build, Git or shared
campaign metadata changes were made. Usage is shared profiling analysis only.
