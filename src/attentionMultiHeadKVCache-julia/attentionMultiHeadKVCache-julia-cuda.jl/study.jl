include("main.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(q,k,v,o,n)
 @cuda threads=(8,8,8) blocks=(1,4) shmem=16896 decode_cuda!(q,k,v,o,Int32(n))
 nothing
end
attention_study(launch!)
