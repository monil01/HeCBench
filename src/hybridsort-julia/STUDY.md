# Hybrid sort: complete source pipeline

## Operator and historical algorithm mismatch

The native benchmark orders Float32 keys using GPUhistogram, CPUadaptivequantile pivots, GPUbucketcount/prefix/scatter, float4 local sort, iterative pairwise merge and final padding removal. All seven logical device stages and the CPU algorithm are present in each new Julia implementation. Historical CUDA.jl delegated to GPU sort and omitted this algorithm; its original entry is preserved in reference_original. A mathematically correct delegated sort is not a faithful source pipeline or comparable execution region.

## Named verified contract

Two independently generated16,384-key fixtures contain positive finite dyadic values with duplicate multiplicities2 and8. DIVISIONS1024 and source launch policies are retained: histogram64×96, bucketceil(N/4096)×32, prefix8×128, float4sort256threads, source adaptive208thread merge with minimum8groups, 2Dfinalpack. Each pipeline executes once per fixture/trial. Warmup runs both complete pipelines before the five checked trials; average trials2–5.

Zeroed host input/output padding is defined by the named fixture. The original random50million-key CLI is unmeasured. The file CLI writes50million values regardless actualfile length and therefore reads beyond its allocation on smallfiles; our adapter directly calls unchanged bucketSort/runMergeSort. This is static source evidence, not an observed sanitizer failure. Negative/nonfinite inputs and uninitialized original padding are not certified. Source histogram maximum-bin wrap and CPU final-bin pivot condition are retained.

Every model's five complete artifacts hold71,828 Float64 exports:32,786 exact input-bit/parameter prefix values,38,914 sorted-key/bucket-metadata values and128 hostcallerbuffer guards. Independent Python comparison-sort and complete multisets check all32,768 output keys per artifact. Every bucket count equals4*size-null, null is0..3, offsets are monotone0..N, and all metadata agrees exactly across models. saved_output_audit.json independently audits20 artifacts/40 fullfixture sort/multiset checks and dependency hashes. Intermediate atomic scatter order is not deterministic and is not claimed identical. Source internal device allocations are not guarded or sanitizer-certified; the captured guards are host buffer suffixes.

## Measured region and ontology

The whole ROI includes allocation, H2D/DtoH, GPUhistogram, dependent CPU pivots/alignment, all bucket/merge/pack launches and frees. It is mixed_host_device_pipeline, not kernel latency. GPUcomponents are CUDA.jl framework_device_kernel and KA/JACC hybrid_device_intrinsics because shared helpers use CUDA-specific block barriers and pointer atomics. KA/JACC provide genuine mapping/launch frontends, with explicit zero dynamic shared memory in JACC. This chosen interoperability is not an unsupported-feature conclusion.

NativecudaMalloc/free versus Julia's pooled CuArray/unsafe_free, additional Julia scratch initialization, host allocations, scalar-tuple versus float4 code generation and launch dispatch are explicit performance covariates. Native bucketSort retains an unfreed small host histogram allocation. No pure abstraction causality, measured bandwidth or roofline claim is made. Total means(milliseconds) are CUDA2.451700, CUDA.jl1.380633, KA1.672245, JACC1.999268, summed over both pipelines.

## Compiler loop and cost

CUDA.jl had three process checks: host parse rejection before GPUJIT, actual merge-kernel InvalidIR from closures capturing reassigned tuples, then five-trial success after source-equivalent unrolled tuple helpers. There are two device-body revisions, two actual device-JIT processes, one host syntax fix and one compiler-feedback repair. Proactive historical algorithm restoration is separately accounted. KA and JACC each have one generic device-body candidate, one successful process/JIT check and zero feedback repairs; each generic kernel specializes seven stages. Stage/grid/merge/trial counts are not attempts. Shared helper/driver reuse, historical unknowns and context/cache-inclusive nonexclusive usage windows are disclosed in effort records.

## Reproduction and checkpoint

study_config.json selects final result20261018T073846061263Z_hybridsort; refs/KA configs preserve sequencing and exact unchanged-series reuse. generate_oracle.py and fixture_provenance.json reproduce four essential CPU input/oracle binaries. reference_history retains parse/IR failures; reference_original retains old delegation. Actual measured output artifacts remain local by hash. Canonical guides and worker checkpoint resume the queue. Root alone stages/commits verified versions separately.
