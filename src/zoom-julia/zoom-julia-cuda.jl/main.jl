using CUDA
include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(a,out,pars,bc,mode,slots)
 H,W,Ho,Wo,pitch,hs,he,ws,we=pars;grid=(cld(Int(Wo),16),cld(Int(Ho),16),bc)
 if mode==1;CUDA.@cuda threads=(16,16,1) blocks=grid shmem=4slots zoom_in!(a,out,pars...,slots)
 else;CUDA.@cuda threads=(16,16,1) blocks=grid shmem=4slots zoom_out!(a,out,pars...,slots);CUDA.@cuda threads=(16,16,1) blocks=(cld(Int(W),16),cld(Int(H),16),bc) zoom_edge!(out,H,W,pitch,hs,ws,hs+Ho,ws+Wo)
 end
end
study((T,n)->CUDA.zeros(T,n),launch)
