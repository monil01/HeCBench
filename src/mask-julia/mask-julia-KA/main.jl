# Mask operators; sourcecopyright/terms in ../LICENSE.
using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","mask_predicate.jl"))
@kernel function mask_kernel!(a,seq,centers,out,m,n,b,radius,stride,mode)
 i=@index(Global,Linear);idx=i-1;total=m*n*b
 while idx<total
  k=idx%m;j=((idx-k)÷m)%n
  @inbounds out[idx+1]=study_keep(mode,j,k,seq,centers,radius) ? a[idx+1] : Int32(-1)
  idx+=stride
 end
end
function study_launch!(a,seq,centers,out,f,mode)
 m,n,barg,b,rad,blocks=f[1:6];stride=blocks*256
 mask_kernel!(KernelAbstractions.get_backend(a),256)(a,seq,centers,out,m,n,b,rad,stride,mode;ndrange=stride)
end
study_main(study_launch!)
