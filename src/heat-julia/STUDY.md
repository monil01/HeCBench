# Heat equation: agreement and discretization

Homogeneous2D diffusion, zeroDirichlet exterior, alpha.1,length1000,totalphysicaltime.5. NO forcing term, despite earlier header-only wording. Native unchanged solve kernel and original genuine CUDA.jl body preserved. Float64initial sine sampled once by deterministic generator; smallessential study_initial.bin+hashmetadata is input, not measured output. All20prefix hashes exact.

Namedgrids63/127/255 with17/18/19steps include odd/even parity. All170246final+previous values checked signed/finite against CPU1e-12, independent discrete sine amplification1e-12 measured GPU(2.34e-15CPU), energydecay and[0,1]maximumprinciple. Full255369artifactvalues include85123inputprefix. ContinuousMMS discretizationerror~1.98e-10,4.95e-11,1.24e-11 remainsdistinct fromportdrift~2e-15. This is scopedaccuracy evidence, not a global convergence theorem.

Composite3solves dividedby54steps; all5correcttimes retained, mean2–5, JIT/setup/reset/H2D/D2H/CPUoracles excluded. Hostlaunch/pointerswap/sync included. Nativeweightedadds vsJuliafactoredneighbors documentedN3rounding. Same256threads and linearcellmapping; JACC distinct precomputedexplicit0Bshared specs perextent, reusing priorFWTrepairknowledge. No GPUsolverlibrary/customintrinsics/crossbackendrun.

Originalsource warning thresholdr.5 is broader than2Dstabilitybound.25; ourcurrent/source-default cases stable. This is staticout-of-scope diagnostic, no failedcurrentkernel or unsupportedfeature. No default1000grid/arbitraryunstableCLI performance claim.

Separateprospective reference/KA/JACC usagewindows observed. HistoricalCUDAjlcostunknown; cached/repeatedcontext, writing andcoordinationmean windowsarenotexclusivebilledcost. Sourcechars/4 keptseparately.
