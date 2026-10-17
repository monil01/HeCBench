# FMA matched FP32 configuration

Named FP32 basic+rowwise only; Float16/BFloat16/Float64 unmeasured. Historical BF16 label actually uses Float32, so noBF16claim.

Native sharedB plus vectorizedbasic/rowwise mapping differs inherited CUDA.jl scalar globalmap/rowwisealias. KA/JACC basic scalar and rowwiseoneoperationperthread; outputrowsunique.

Native exactFP32 devicebodies excerpted with hashes because fulloriginalheader fails unrelatedBF16specialization on CUDA12/sm70. Onecompatibilityrepair plus one restrict-declaration instrumentationrepair, zeroKA/JACCcompilerrepairs.

Full262144Float32 outputs include untouchedzero rows; all5 bitwiseequal bothrefs+independentCPU on signed dyadic nontrivial gather/scatterfixture. Zero fills+bothkernel launches inside each repeat allmodels.
