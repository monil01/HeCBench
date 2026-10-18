# collision: duplicate keys within a warp

## Scientific and numerical scope

The source detects equal integer keys within one32-lane warp and returns a mask that excludes the first original lane for each key. It is not a geometric collision benchmark. A CPU existential-equality oracle, set cardinality and per-key minimum-origin representatives independently agree with the complete GPU outputs. Forty fixtures cover every source duplicate level and signed, extreme and scattered groups.

All five trials in all four models agree bit-for-bit:4522 UInt32 words per artifact contain1282 input/parameter words,1280 real flags,40 masks and1920 actual device input echo words. The input echoes include640 real guard cells per trial. Unchanged native output globals contain32 flags and one mask; they have no output canaries, and no output-guard guarantee is invented. The input fixture and every explicit dependency, including binary/JSON files, are bound by the new opt-in runner hash protocol.

The source comment suggesting mask|(mask>>1) reconstructs all participants fails after origin-lane restoration (a duplicate only at lanes0 and31 is a concrete witness). The tested operator is the source's serialization mask, consistent with its actual tests. No full-participant-mask claim is made.

## Mapping and evidence

Historical CUDA.jl kernels genuinely compute the same operator using a redundant quadratic scan and serial mask. They are archived intact; the new reference restores the native warp-bitonic mapping without claiming an observed historical numerical failure. All three Julia GPU versions use one reference-authored, audited network helper and actual CUDA shuffle/vote intrinsics, with genuine CUDA.jl/KA/JACC kernel dispatch and indexing. These are explicitly backend-specific hybrids. A workgroup-shared portable alternative was not authored or probed; no framework unsupported claim follows.

Each current GPU candidate passed its first compiler/check invocation. Shared helper effort is attributed once to reference preparation; subsequent framework wrapper windows are marginal ordered costs. Historical translation effort remains unknown.

## Performance and limitations

Final evidence: `.porting-state/portable_julia/runs/20261018T044953441527Z_collision`. Mean trials2–5 microseconds: CUDA 294.435070, CUDA.jl 540.975495, KA 583.818672, JACC 702.532200. Each trial sums40 regions of100 repeated two-kernel launches, divided100; per-case durations are retained. Allocation, transfers, random generation, reset, capture and warmup are excluded. This is a named kernel component study, not source CLI elapsed time.

Native static output symbols and Julia passed arrays differ. Native network stages are explicitly unrolled while Julia uses a fixed stage tuple; compiled instructions, registers, occupancy and dispatch/device contributions remain unprofiled. No intrinsic framework winner, bandwidth limit or roofline location is inferred from these host-inclusive microbenchmarks.
