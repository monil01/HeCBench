using CUDA
import JACC
JACC.@init_backend
include("study_kernel.jl")
const TEMPLATE=JACC.zeros(Float32,129)
function launch(n,args...)
 spec=JACC.launch_spec(threads=128,blocks=n,shmem_size=516,sync=false)
 JACC.parallel_for(spec,128n,binomial_kernel!,args...,TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
