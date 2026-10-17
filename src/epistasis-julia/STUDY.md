# epistasis: synthetic genetic interaction screening

The actual kernel counts two phenotype-specific3x3 genotype tables using packed UInt32 bitsets and popcounts, then evaluates a Float32 approximate log-factorial score for every SNP pair. The historical Julia arithmetic was genuine. Its named adapter now matches the native padded2D launch geometry. KA and JACC retain one thread per pair, private count tuples and the same score formula; all three are framework device kernels without CUDA-specific device intrinsics.

## Scientific and numerical evidence

A generated135-subject67-SNP fixture plants phenotype as the XOR of SNP0 and SNP1. The two phenotype groups contain67 and68 subjects, exercising both masked partial final packs. An independent oracle counts raw genotype/phenotype values directly and independently repacks inputs using explicit bit positions. All2211 computed pair scores and2278 untouched matrix sentinels are checked in all20 model/trial captures, alongside9992 exact serialized input values. Pair0,1 is ranked first in every capture. Mutual information in bits is {"snp0": 4.0173458059857524e-05, "snp1": 4.0173458059857524e-05, "pair0_1": 0.9999604195271727}; this illustrates a pairwise synthetic interaction, not clinical/statistical calibration.

Device arithmetic error is measured against the Float64 evaluation of the unchanged source formula at the original absolute1e-3 tolerance. The source's Stirling-like approximation differs from exact log-factorials by up to0.57176972 on this fixture. That model approximation is separate from port arithmetic drift. Actual GPU contingency arrays are not separately exported, so the evidence is the full native score matrix plus independent raw-count oracle.

## Preserved failures and repairs

The first native instrumentation candidate passed phenotype masks in reversed order, despite a correct serialized prefix. The raw-genotype oracle caught error1.778375; the launch arguments were corrected without changing the kernel or threshold. This demonstrates why prefix-byte equality alone cannot establish actual input use. The first CUDA.jl full-output check then exposed `typemax(Float32)` infinity versus native finite `FLT_MAX`; `floatmax(Float32)` repairs sentinel encoding without changing computed score arithmetic. The failed log contains an invalid JSON Inf metric and no retained validated trial artifact; that capture limitation is explicit. Historical argmin-only validation could miss this marker mismatch.

Five trials per model retain all outputs; timings average trials2–5, each a ten-kernel batch mean. Allocation, packing, reset, transfers and oracle work are outside timing. Mean microseconds: {"cuda": 7.271575, "cuda_jl": 9.241075, "ka": 10.857575, "jacc": 20.93145}. Result: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T223221226778Z_epistasis/result.json`. No profile establishes a bottleneck or framework causal mechanism. Prospective activity windows include cached/repeated context, same-benchmark witness/metadata work, coordination and waiting, not exclusive coding or billed cost; partial source-size proxies remain separate.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
