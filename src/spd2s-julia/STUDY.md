# spd2s: scientific and implementation observation

The scientific operator is lossless dense-to-CSR conversion. The benchmark name does not describe an SPD solve. Native cuSPARSE analysis computes row offsets and the nonzero count before timing. All four models run that same analysis, then compare native vendor conversion with real Julia device kernels that scan one row per work item. Column–value pairs stay together during per-row canonicalization, and the resulting CSR matrix must reconstruct every dense input entry exactly. The original checker sorts all column indices and values independently; a minimal counterexample shows its blind spot without establishing a failure in the native conversion. Convert-only timings exclude analysis and allocation, so they cannot estimate the cost of complete application format conversion.

## Reference audit

Historical Julia runs correctCPU init_csr loop andcompareswithsameCPU routine, withnoCUDAdevicework. This is backend/execution mapping absence, notwrongscientificconversionoperator. Preserve bytes andimplement genuineGPUtimedconversion. Native globallysortscolumns/values independently, losingpairing; newchecker retainsper-rowpairs andfullreconstruction.

## Numerical and performance evidence

Configuration `spd2s_dense_to_csr17x19_65x33_analysis_outside_convert_only_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T234309070633Z_spd2s/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One authored candidate and one device JIT validation process per Julia model; no target compiler, runtime or numerical repair. The shared native measurement harness first missed the iostream header, causing a std::cout lookup error. That failed build is preserved as shared instrumentation effort. Adding the required header left the native algorithm and measured CUDA.jl source unchanged, allowing reuse of the original CUDA.jl capture. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/spd2s-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Scientific witness and staged timing scope

All 20 saved captures pass an independent complete CSR-to-dense reconstruction. Every row offset matches separately counted nonzeros; paired columns and values reconstruct all 2,468 dense entries exactly. Empty and full rows, negative values, and positive/negative zero are included. All 1,322 output values and 3,796 input/derived-analysis prefix values are checked per capture. Canonicalization sorts column–value pairs only within each row.

The legacy native checker sorts columns and values independently across the matrix. A two-entry example ([1,2] versus swapped [2,1] values with the same columns) passes those separate multiset checks while changing the matrix. This is a checker blind spot, not an observed native conversion failure; the stronger paired reconstruction avoids it.

The timed implementation class of each Julia version is framework_device_kernel, with one authored row-compaction kernel and no device-specific intrinsics. The complete program also calls a CUDA-specific cuSPARSE analysis adapter outside timing. That preparation dependency is explicit; no claim that this setup is portable across backends. Native conversion uses the vendor library, so the comparison includes an algorithm-mapping difference. cuSPARSE 12001 (/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140) is shared for all analysis. Context-wide completion includes raw vendor default-stream work.

One authored candidate and one device JIT validation per Julia model. A missing iostream header caused one failed shared native harness build, preserved separately from target translation effort. No target compiler, runtime or numerical feedback repair was observed. Original source and original CPU-only Julia reference are preserved.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
