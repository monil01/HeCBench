using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,out,atoms,L)
 @inbounds out[i]=factor(sample(i-1,L)...,atoms,L,Val(false))
end
run!(out,atoms,L)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=16,shmem_size=0,sync=false),4096,kernel!,out,atoms,L)
study(run!)
