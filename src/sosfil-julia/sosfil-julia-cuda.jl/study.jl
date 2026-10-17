include("main.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(signals,samples,::Val{SHARED},sos,zi,x,template) where SHARED
 @cuda threads=64 blocks=signals shmem=SHARED*sizeof(eltype(x)) sosfilt_cuda!(sos,zi,x,samples,Val(SHARED))
 nothing
end
sos_study(launch!)
