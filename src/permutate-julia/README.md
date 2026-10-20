# permutate: four-way source-behavior study

The original DVC million-sample file, 2500 streams, ten250-thread blocks, source early-stop/controller and CPU compression stage are retained. The actual source default is size8 despite the one-bit filename. The original fixture and both bounded controls stop after oneGPUbatch (2500 logical stream evaluations); do not claim10000 distinct permutations. Four source GPU definitions are preserved byte-identically. Each Julia version authors four genuine GPU entries through its own dispatch. Native allocation/controller/CPU compression remain a scoped C ABI hybrid; selected native GPU fallback is disabled.

**Scientific qualification:** the source shuffle excludes self-swaps, restarts the same streams everybatch, and omits terminal-run maxima. Independent finite/source CPU witnesses demonstrate the defects. This is exact numerical source-behavior agreement, NOT a NIST conformance, IID or randomness certificate. The binary CPU compression string allocation additionally needs one extra terminator byte (ASan witness and corrected pass). This conditional reference and known-scientific-defect stratum is excluded from untouched-native/scientific-performance recommendations. See reference_provenance.json and scientific_witness.json.

## Checked contract

All20 complete observable artifacts (1010827UInt8 bytes/trial) compare exactly to BOTH references: actual input prefix1007356bytes, all original19 baseline statistics,57 final ranks, IID decision, complete GPUcounter histories and CPUcompression-stat histories. No value canonicalization or tolerance relaxation. Every full2.5-billion-element primary internal shuffle is independently checked on everytrial against a separate complete CPU permutation/statistic oracle. Per-thread GPUstatistic temporaries are not individually emitted by the source; the independent oracle recomputes18 statistics per2500streams and checks all observable rank counts.

One complete packed internal shuffle percase/model at measuredtrial1 is retained; the other four full internal arrays are checked but NOT separately archived. Packing is injective only for binary-valued data; the full eight-bit control retains every raw byte. The GPU matrix itself remains full2.5GB UInt8, never bitpacked for computation. Identical independently acquired captures shareinodes after completeSHA+bytecomparison; everyoriginal path/content and acquisition metadata are retained (internal_capture_storage_review01–03.json). No unique data or other-worker evidence was removed.

Own16-byte GPUallocation guards and readonly buffers are checked; these are added study guards, not original CLI guarantees. Support is the named file and bounded2048-value eight-bit/4096-bit controls, binarylength divisible8. No universal near-tie/statistical-distribution guarantee.

## Timing and mappings

Primary region is the original GPU shuffle/statistics/counts-D2H/CPUearlystop loop plus common counter-history copies and completionfence. CPUinitial statistics/compression, allocations, readbacks, independent checks, packing and downstream CPUcompression are outside this ROI. The full source CPUcontroller still executes eachwarm/trial. Auxiliary `full_controller_wall_seconds_auxiliary` INCLUDES all added science observations and serialization: it is NOT clean originalCLI end-to-end time or a CPUfraction estimate. Hash-bound metadata-only clarifications in study_adjudications.json preserve everyraw row/time/verdict.

| Model | Mean GPU-region seconds (trials 2–5) |
|---|---:|
| cuda | 2.804193518 |
| cuda_jl | 2.979970943 |
| ka | 2.970934681 |
| jacc | 2.969681405 |


Compilewarmup precedes five checked trials; average2–5, no outlier removal. Native-only firstaccepted series2.7970227075s is preserved. A wrong reuse setting caused an extra native series2.804193518s beforeCUDAjl; subsequent references are reused strictly. Its inclusion follows execution provenance, not speed selection (native_extra_series_disposition.json).

AllJulia versions use the same accepted equations. Collision membership source256Bool becomes4UInt64; binaryhamming uses popcount and indices are Int64. These are mapping/performance covariates; register/spill/occupancy and roofline are unmeasured. KA uses real @kernel/globalindices; JACC uses real1Dparallel_for, cached complete250-thread/10,20,40-block specifications, shmem_size0 and legacy stream. Both use explicit CUDA UInt32atomic extension. No backend-independent capability or framework-unsupported inference.

## Reproduce from a clean checkout

Use existing nvcc12.0, Julia1.12.6 and pinned Project/Manifest/preferences (pinned_environment.json). No system package changes or sudo. From repositoryroot:

```bash
python3 src/permutate-julia/retrieve_assets.py
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/permutate-julia/build_program.py --mode native --output /tmp/permutate_prepare --arch sm_70
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/permutate-julia/bounded_native_prepare.py /tmp/permutate_prepare
python3 src/permutate-julia/run_study.py --models all
```

Preparation requires an EMPTY canonical directory and refuses overwrite; it generates actual GPUinput readbacks and independently computed CPUexpected permutations/statistics only, never accepted trial rows. About321MB of regenerable canonical/oracle data, a~2.5GB host CPUworkspace and up to~1.3GB retained internal captures without hardlink consolidation are needed. Per-file400MiB and2GiB free-space limits apply to preparation. Do NOT rerun accepted measurements to regenerate documentation. The centralverifier owns GPU6lock; never add an outer flock to run_study.py.

Original archive155139B: MD5 `69ad908d5373a3a0732eda45b7b79577`, SHA256 `05268cb94ae9a1b61ad0067fc050d34c5893032baa780605cecb2846c33db22c`. Member1M SHA256 `f9ea8832af4c4205f518845b264465800921688fc2c4d566fbc087664aeb2313`. HTTPavailability review is a separate currentHEAD observation; originalGET status was not retained and is not backfilled.

Expected canonical SHA256:

`actual_input_case0.bin` ef37a3586885bc41290b617ebb943c3fbcb457bf73fed351201455be5f111d8f

`actual_input_case1.bin` b6a6c22c4ffeea548cd6078b2f261ce18f96942ccfa398671008c845c3c8c9d8

`actual_input_case2.bin` 95f1a6945aa3ea19e2d0a8227dc94d1a9aa46634fb5bf7eb5ddeac946092f61b

`independent_full_shuffle_and_stats_case0.bin` 5cb62cde1c7fd030a2b9d2542716222064bbb120e39d5a69cea6bf81c975b837

`independent_full_shuffle_and_stats_case1.bin` ebc866b882aadd210534d3cb1d766ad488e9449b382b6001799548a83d274785

`independent_full_shuffle_and_stats_case2.bin` 8ecdfab3e37551c694bee08ad6c7ad162c290dbd198129baf0c9738895d4527c


CPU-only complete saved-artifact replay:

```bash
python3 src/permutate-julia/audit_captures.py --result PATH_TO_RESULT_JSON --output /tmp/permutate_cpu_audit.json
```

Reproduce the small executed CPUallocation witness with existing g++/ASan (original is expected to report overflow, corrected exits0):

```bash
g++ -O1 -g -fsanitize=address -fno-omit-frame-pointer src/permutate-julia/compression_allocation_witness.cpp src/permutate-julia/reference_history/statistical_test.cpp src/permutate-julia/reference_history/bzip/*.cpp -o /tmp/permutate_original_asan
/tmp/permutate_original_asan
g++ -O1 -g -fsanitize=address -fno-omit-frame-pointer src/permutate-julia/compression_allocation_witness.cpp src/permutate-julia/configured/statistical_test.cpp src/permutate-julia/reference_history/bzip/*.cpp -o /tmp/permutate_corrected_asan
/tmp/permutate_corrected_asan
```

Actual inclusive usage counters, context reuse/interleaving and candidate units are in effort_summary.json. Native preparation, target translation, shared verification and annotation windows are separate; do not sum overlapping phases or interpret cached-inclusive counters as intrinsic framework costs. CUDAjlwindow contains accidental native work; JACCwindow includes storage planning for future queue. Allthree are excluded clean-target cost calibration. RQ1–5 facts/limits: paper_observations.md/JSON. Large regenerated binaries/captures stay outside sourceGit.
