# Native is cuSPARSE-only: this version orchestrates that same vendor API.
# No JACC device kernel is authored or claimed for this source benchmark.
using CUDA,JACC
include("../study_common.jl")
sddmm_study()
