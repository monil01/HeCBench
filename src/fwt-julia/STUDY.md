# FWT study

## Mathematical and scientific scope

The source demonstrates naturally ordered Walsh-Hadamard transforms for **dyadic (XOR-index) convolution**, y[i]=sum_j x[i XOR j]k[j]. This differs from ordinary spatial shift convolution. Electrical-engineering/numeric-theory transform use is explicit in source comments; no particular scientific dataset or physical signal is identified.

The named configuration has65536 Float32 data values and128 nonzero kernel values, zero-padded to65536. Nonconstant dyadic fixtures are identical across all4 models. One timed operation consists of forward(data), forward(kernel), normalized pointwise modulation, and inverse Walsh transform. Input reset/H2D and all validation/readback are outside timing. There is one complete pipeline pertrial, explicit device JIT warmup and5 synchronized in-process trials; the mean uses trials2–5.

## Numerical contract

All65536 outputs are checked against an independent direct CPU XOR convolution accumulating in Float64. Every output and both references agree exactly on this fixture. The original native relativeL2<1e-6 criterion is also enforced; elementwise tolerance is1e-5. This strengthens the inherited Julia main's relaxed1e-5 L2 criterion. General non-dyadic inputs can round differently due butterfly order; exact fixture agreement does not imply universal bitwise equality.

## Mapping and interpretation

The existing CUDA.jl port is genuine, and its device kernels are unchanged apart from a host entrypoint guard. Native CUDA uses3 global radix4 passes and a4096-byte shared radix4 transform per Walsh transform:13 kernels per full convolution. CUDA.jl/KA/JACC inherit16 global radix2 stages pertransform:49 kernels total, no shared scratch. Native modulation launches128 blocks with2 items perlane; Julia-family launches256 blocks with one. These explicit algorithm/mapping differences confound attributing the entire performance gap to framework overhead. No profiling supports a specific bottleneck claim.

The two device arrays occupy512KiB and fit V100 L2. Native source default log2N23 (64MiB pair) and its random input distribution are unmeasured. This result does not establish DRAM or application-scale behavior.

## Compiler loop and launch repair

KA passed its first program candidate/check. JACC's first program compiled but failed the original numerical criterion (relativeL2=17572.799). Pinned JACC1.4 mutates LaunchSpec.blocks at first use. One spec first handled32768 stage items (128blocks) and then65536 modulation items, incorrectly retaining128blocks and leaving half the modulation unlaunched. Candidate2 uses separate explicit stage128/modulation256-block specs, bothshmem_size=0, and passes all outputs. The device callbacks/arithmetic are unchanged:2 program candidates/checks,1 device-kernel candidate and1 numerical/host-launch-policy repair. The failed candidate source and backend source hash are retained. This is an agent launch-API error, not unsupported mathematics.

A reference-metadata Boolean assertion failed after successful verification and was repaired without changing/rerunning kernels; this shared tooling event remains separate from target effort. Original CUDA.jl generation effort is unknown. Prospective consumed-usage windows include cached/repeated context, shared work and coordination; partial source proxies stay separate. Broad workflow overlaps phases and is not additive or monetary billing. JACC intentionally starts with successful KA mapping available.
