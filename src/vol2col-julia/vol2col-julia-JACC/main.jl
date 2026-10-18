using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(k,vol,col,pars)
 spec=JACC.launch_spec(threads=512,blocks=1,shmem_size=0,sync=false)
 if k==1;JACC.parallel_for(spec,512,vol2col_jacc!,Int64(210),vol,pars...,col,Int64(512))
 else;JACC.parallel_for(spec,512,col2vol_jacc!,Int64(210),col,pars...,vol,Int64(512))
 end
end
study((T,n)->JACC.zeros(T,n),launch)
