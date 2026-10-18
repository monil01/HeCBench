using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function log_jacc!(i,input,out,n::Int32,row::Int32,p::Int32)
 if i<=n;@inbounds out[1+row*n+i]=binary_log_study(input[i],p);end
 return
end
function factory(input,out,n)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 (row,p)->JACC.parallel_for(spec,n,log_jacc!,input,out,n,row,p)
end
log_study(factory)
