using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=JACC.get_backend(:cuda)
const SPEC=JACC.LaunchSpec{typeof(BACKEND)}(threads=256,blocks=cld(N,256),shmem_size=0,sync=false)
@inline function encode_study!(i,dst,src,n,fmt,sat)
 if i<=n
  @inbounds x=src[i]
  @inbounds dst[i]=sat ? fp8_sat(x,fmt) : fp8_ref(x,fmt)
 end
 return
end
function launch!(path,dst,src,n)
 JACC.parallel_for(encode_study!,SPEC,n,dst,src,n,Int32((path-1)÷2),iseven(path))
end
study_main(launch!)
