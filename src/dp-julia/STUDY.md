# Dot-product paths and loaded-library provenance

Three alternatives are separate configurations: source custom CUB block reduction/global atomic, actual vendor cuBLASDotEx, and full transformed reduction returning a host scalar (native NVIDIA Stdpar GPU). All preserve four cases: n4099/padded5120/grid1 and n16387/padded17408/grid4, each FP32 and FP64. Original268435456-element/1000-repeat CLI remains unmeasured. Source device body is unchanged; native compiler is NVC++25.5 with Stdpar GPU/CUDA12.9, including source warnings about unsupported CUDA-C++ compilation. Warnings are not failed evaluations.

## Inputs, arithmetic and complete outputs

Original std::mt19937(19937)/uniform_int_distribution(-32,32) interleaved A/B draws are serialized by a C++ generator. G++ and actual NVC++25.5 generator replays are byte-identical. Input readback from the actual GPU buffers supplies90120 losslessly Float64-encoded prefix values; each trial also records all4 scalar DOT outputs. Expected answers−21727/−73477 repeat for FP32/64. Independent integer products/sums and actual sum-absolute-product bounds1081879/4254127<2^24 prove exact FP32/64 results under any reduction order on these fixtures. All60 primary model/trial records plus20 control records pass complete prefix/scalar checks. This does not certify arbitrary real-valued conditioning, sourcehuge sizes or all internal intermediates.

## Mapping and timed regions

Custom variants use4contiguous products/group, source grid stride,1024thread blocks and1atomic/block. Native CUB and Julia shared halving trees differ. JACC additionally copies1024 template values and synchronizes before overwriting shared scratch. Custom reset is inside each repeat: native cudaMemset versus Julia cuMemsetD8Async on the kernel task stream. Vendor paths have0authoreddevicekernels and use actual DotEx/device scalar results. Julia cuBLAS handles are source-audited to bind currenttaskstream and devicepointermode; all timed boundaries block device-wide.

The scalar-return idiom includes implementation allocations and host scalar return: CUDA.jl materializes a GPU product then sums; KA explicitly launches its reduction and reads a scalar; JACC uses public parallel_reduce with actual Float32/Float64 type, two generated reduction kernels and scratch allocations. Native Stdpar GPU internals are opaque. These are matched operator/region comparisons with declared algorithm/API/materialization differences, not isolated abstraction penalties. Allocation of primary vector inputs, capture and oracle stay outside.100warmup calls precede five trials; each trial sums four20-repeat loop means. Discard only timing1.

## Actual libraries and separate control

The full link argument naming local cuBLAS12.0 did not pin the native loaded DSO: NVC RUNPATH resolved soname12 to SDKcuBLAS12.9, confirmed by CPU-only linkage audit and trial API version120900. Julia actually loaded12.0.2.224/version12.0.2/runtime12.0. A function-address dladdr printed the executable PLT stub, not the implementation DSO. Hash-bound metadata corrections preserve the mistaken raw prose and every numerical/timing capture. Library/compiler/runtime differences are explicit covariates.

Before measuring a control, native cuBLAS/Lt12.0 preloads were declared, primary exclusion set, and all unchanged Julia model specs/dependencies/inputs/regions were checked for exact reuse. Native fresh five trials report120002 and pass every scalar/prefix; its NVC/runtime12.9/source/flags remain unchanged. Both coupled DSOs changed; the sequential unrandomized series with four retained samples does not estimate a general causal abstraction effect. The original12.9 primary remains regardless of speed.

## Effort

One current program candidate per model, three path-process evaluations. CUDA.jl custom scientific kernel JIT occurs in one process, KA uses it in custom and scalar-return processes, JACC compiles custom once and the public-reduction callback separately; generated library kernels are not newly authored kernel definitions. Two dtypes/cases/five trials are separate units. No framework compiler/runtime/numerical repairs observed. Historical counts/costs unknown. Shared helper/oracle/restoration and library-provenance instrumentation belong to reference work; control has a separate phase. Prospective activity windows include cached/repeated context, guide/paper work, other source commits and coordination, and overlap nonadditively.

The shared annotation/workflow END markers were captured in the same tool invocation as their final metadata refresh. These activity spans have explicitly partial coverage; tail consumption is unknown. Counter deltas remain unchanged, and usage_scope_adjudications.json preserves the boundary deviation. KA/JACC translation windows are separate; no complete exclusive translation-cost claim is made.
