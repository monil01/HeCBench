using CUDA,JACC
JACC.@init_backend
function makewords(a)
 d=JACC.array(a)
 d isa CUDA.CuArray{UInt32} || error("JACC backend allocation must be device UInt32 storage before cuSPARSE pointer interoperation")
 d
end
include("../study_common.jl")
spmm_study(makewords)
