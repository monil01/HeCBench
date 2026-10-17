using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function encode_study!(dst,src,n,fmt,sat)
 i=@index(Global,Linear)
 if i<=n
  @inbounds x=src[i]
  @inbounds dst[i]=sat ? fp8_sat(x,fmt) : fp8_ref(x,fmt)
 end
end
const ENCODE=encode_study!(CUDABackend(),256)
function launch!(path,dst,src,n)
 ENCODE(dst,src,n,Int32((path-1)÷2),iseven(path);ndrange=n)
end
study_main(launch!)
