using CUDA
import JACC
JACC.@init_backend
# Native source is vendor SGEMV with host RBF, not an authored device kernel.
# Actual JACC allocation integrates explicitly with CUDA's vendor BLAS API.
include("../study_common.jl")
runstudy(JACC.array)
