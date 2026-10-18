using CUDA,JACC
JACC.@init_backend
function makewords(a)
 d=JACC.array(a)
 d isa CUDA.CuArray{UInt32} || error("JACC backend device storage required before cuSPARSE pointer interoperation")
 d
end
include("../study_common.jl")
sps2d_study(makewords)
