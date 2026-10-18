using CUDA,KernelAbstractions
function makewords(a)
 d=KernelAbstractions.allocate(CUDABackend(),UInt32,(length(a),))
 copyto!(d,a)
 d
end
include("../study_common.jl")
sps2d_study(makewords)
