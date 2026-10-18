# Native is cuSPARSE-only: this version orchestrates that same vendor API.
# No KA device kernel is authored or claimed for this source benchmark.
using CUDA,KernelAbstractions
include("../study_common.jl")
sddmm_study()
