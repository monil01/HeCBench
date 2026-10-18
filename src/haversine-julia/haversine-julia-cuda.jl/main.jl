using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function hv_cuda!(a,b,c,d,out,n::Int32)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n;@inbounds out[i+1]=hv_value(a[i+1],b[i+1],c[i+1],d[i+1]);end
 return
end
function factory(inputs,out,n)
 a,b,c,d=inputs
 ()->(@cuda threads=256 blocks=cld(n,256) hv_cuda!(a,b,c,d,out,n))
end
haversine_study(factory)
