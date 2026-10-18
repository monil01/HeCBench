using CUDA
function vectoradd!(a,b,c,n)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n;@inbounds c[i]=a[i]+b[i];end
 nothing
end
launch(a,b,c,n,s)=(@cuda threads=256 blocks=cld(Int(n),256) stream=s vectoradd!(a,b,c,n);nothing)
include("../study_common.jl")
allocation_study(launch)
