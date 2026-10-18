using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function hv_jacc!(i,a,b,c,d,out,n::Int32)
 if i<=n
  @inbounds out[i+1]=hv_value(a[i+1],b[i+1],c[i+1],d[i+1])
 end
 return
end
function factory(inputs,out,n)
 a,b,c,d=inputs
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,n,hv_jacc!,a,b,c,d,out,n)
end
haversine_study(factory)
