using CUDA,KernelAbstractions
# Source is host runtime allocation/free only: no KA device kernel exists.
# Explicit CUDA runtime interoperability; do not substitute framework memory pools.
include(joinpath(@__DIR__,"..","study_common.jl"))
mallocfree_study()
