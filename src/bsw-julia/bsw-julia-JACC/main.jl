using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
function sequence_jacc!(i,inverse,a,b,oa,ob,ba,ea,bb,eb,scores,open,extend,scoring,encoding)
 bsw_device!(inverse,a,b,oa,ob,ba,ea,bb,eb,scores,open,extend,scoring,encoding)
 nothing
end
function launch_jacc!(inverse,threads,args...)
 spec=JACC.launch_spec(threads=threads,blocks=N,shmem_size=SHMEM,sync=false)
 JACC.parallel_for(spec,N*threads,sequence_jacc!,inverse,args...)
 nothing
end
bsw_study(launch_jacc!;allocate=JACC.array)
