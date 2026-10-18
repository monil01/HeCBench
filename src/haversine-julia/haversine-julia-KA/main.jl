using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function hv_ka!(a,b,c,d,out,n::Int32)
 i=@index(Global,Linear)
 if i<=n
  @inbounds out[i+1]=hv_value(a[i+1],b[i+1],c[i+1],d[i+1])
 end
end
function factory(inputs,out,n)
 a,b,c,d=inputs
 k=hv_ka!(CUDABackend(),256)
 ()->k(a,b,c,d,out,n;ndrange=n)
end
haversine_study(factory)
