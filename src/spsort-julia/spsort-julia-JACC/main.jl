using CUDA,JACC
JACC.@init_backend
function makewords(a)
 d=JACC.array(a)
 d isa CUDA.CuArray{UInt32} || error("JACC storage must be device UInt32 before cuSPARSE interoperation")
 d
end
include("../study_common.jl")
spsort_study(makewords)
