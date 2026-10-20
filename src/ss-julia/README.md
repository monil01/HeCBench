# SS: complete source text search

The original AMD string-search algorithm searches the complete 707,706-byte OpenCL 1.2 specification for all case-insensitive overlapping occurrences of `clEnqueueNDRangeKernel` (22 bytes), for the original 20,000 GPU repetitions. The text is input data, not an OpenCL runtime dependency. Six full-field controls exercise naive search, dense overlapping results, both load-balance filter stages, cross-group boundaries, mixed case and the last valid start.

The native GPU file is unchanged. The named reference initializes otherwise undefined output padding and uses overlap-safe host compaction. An original-source initcheck observed 88,458 inactive-result read errors; a separate legal dense-results CPU ASan witness establishes overlapping `memcpy`. Neither is evidence of incorrect original valid GPU match positions. This conditional reference is excluded from untouched-native aggregates. See `reference_provenance.json` and `host_compaction_witness.json`.

Numerical agreement means **all group counts and all source-final sorted match positions**. The source itself compacts and sorts before comparison. Atomic append order is not claimed identical. All physical before/after allocations, inactive slots, guards, complete host tail and a complete canonical-to-physical slot permutation are retained and independently replayed byte for byte. This is not a count-only property check. Each primary artifact contains 5,533,660 UInt32 words, including 4,970,457 exact actual-input/guard words. No floating error budget applies.

## Clean checkout reproduction

Run from the repository root. Existing pinned Julia projects, nvcc and OpenSSL development headers are required. These commands do not install system packages. Use a **fresh** fixture directory; fixture/preparation generators refuse to overwrite an existing campaign.

```bash
SS_FIXTURES=/tmp/hecbench-ss-reproduction
python3 src/ss-julia/retrieve_asset.py --output "$SS_FIXTURES"
python3 src/ss-julia/prepare_controls.py --output "$SS_FIXTURES"
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/ss-julia/build_program.py --mode native --output /tmp/hecbench-ss-native --arch sm_70
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 /tmp/hecbench-ss-native "$SS_FIXTURES" /tmp/hecbench-ss-reproduction-private 1
python3 src/ss-julia/make_config.py --fixtures "$SS_FIXTURES" --models cuda,cuda_jl,ka,jacc --output /tmp/hecbench-ss-all4.json
python3 src/ss-julia/run_study.py --config /tmp/hecbench-ss-all4.json
```

The central runner owns the GPU lock; do not wrap its invocation in another flock. Its final printed canonical result path is the input to CPU-only full replay:

```bash
python3 src/ss-julia/audit_captures.py RESULT_JSON --fixtures "$SS_FIXTURES" --output /tmp/hecbench-ss-replay.json
```

Asset archive: 148,894 bytes, MD5 `57b2a3c204f0d296709d4792ef403f76`, SHA256 `cc74819fb3c522e37046d4f190dc549c2c5f6edf0793702f7f919c75f5c73959`. Extracted text SHA256 `d6e6cac809f721d161ff53e77c0630eca066ec6ff6149ffb01f9c099488d2c87`; expected primary matches 36 with group counts `[3,0,17,10,1,5]`. `cases.json` records all generated text/oracle hashes and expected control counts. Regenerable arrays, raw trials and compiled programs stay outside source Git.

## Mapping, timing and retained failures

All targets have their own two authored GPU bodies and pure Julia host BMH/compaction/control. KA dispatch retains CUDA thread indices, block barriers and shared atomics as explicit backend extensions; its pattern bytes use Val-specialized static shared storage. CUDA.jl/JACC retain dynamic pattern storage. JACC uses complete `(mode,groups,pattern_bytes)` launch-spec keys, 256 threads and exactly the source dynamic shared bytes. No native GPU fallback or C ABI is used by these targets; unused optional C ABI code in the shared native adapter is not invoked.

The ROI includes the complete source repeat loops, host launch control and whole-context completion; input allocation, copies, CPU oracle, capture and source-final CPU sort are outside. One complete warmup precedes five checked trials; report the average of trials 2–5 with all five retained. This is not full CLI wall time or pure GPU event time. Targets use one owned explicit CUDA stream and blocking context completion after a pinned CUDA package legacy-stream synchronization worker crashed in an incomplete series. The kernel bodies passed warmup and two checked trials before that crash; its captures and the earlier host JSON parse failure remain local diagnostics. No failed timing is substituted for a five-trial series.

`all4_full_capture_audit.json`, `final_dependency_review.json`, `all4_accuracy_timing_summary.json`, `effort_summary.json`, `domain_snippet.json` and `paper_observations.md` bind final evidence. Local `/tmp` data and canonical symlinks require archival before reboot or cleanup. Token counters describe inclusive observed ordered-workflow windows with inherited/cached context, not exclusive billing or intrinsic framework costs.
