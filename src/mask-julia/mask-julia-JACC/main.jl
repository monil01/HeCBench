# Sourcecopyright/terms in ../LICENSE.
using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","mask_predicate.jl"))
@inline function mask_callback!(i,a,seq,centers,out,m,n,b,radius,stride,mode)
 idx=i-1;total=m*n*b
 while idx<total
  k=idx%m;j=((idx-k)÷m)%n
  @inbounds out[idx+1]=study_keep(mode,j,k,seq,centers,radius) ? a[idx+1] : Int32(-1)
  idx+=stride
 end
end
function study_launch!(a,seq,centers,out,f,mode)
 m,n,barg,b,rad,blocks=f[1:6];stride=blocks*256;backend=JACC.get_backend(:cuda)
 spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(mask_callback!,spec,stride,a,seq,centers,out,m,n,b,rad,stride,mode)
end
study_main(study_launch!)
