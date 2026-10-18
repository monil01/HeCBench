using CUDA,KernelAbstractions
function makewords(a)
 d=KernelAbstractions.allocate(CUDABackend(),UInt32,(length(a),))
 copyto!(d,a)
 d
end
include("../study_common.jl")
spsort_study(makewords)
