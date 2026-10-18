using CUDA
import JACC
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function bond_kernel(i,curve,repo,current,maturity,clean,bond,strike,dirty,acc,outclean,yields,n)
 bonds_body!(curve,repo,current,maturity,clean,bond,strike,dirty,acc,outclean,yields,n,Int32(i-1))
 nothing
end
function launch(dev,n)
 blocks=cld(n,256)
 spec=JACC.launch_spec(;threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,bond_kernel,dev...,Int32(n))
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(JACC.array,launch)
