# Sourcecopyright/terms in ../LICENSE.
using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","md5_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function search_kernel!(fi,fk,fd,n,len,base,s0,s1,s2,s3)
 i=@index(Global,Linear)
 study_search_thread!(i-1,fi,fk,fd,n,len,base,s0,s1,s2,s3)
end
@kernel function probe_kernel!(w0,w1,lens,out,n)
 i=@index(Global,Linear);study_probe_thread!(i,w0,w1,lens,out,n)
end
function search_call!(f,hi,hk,hd)
 fill!(hi,Int32(-1));fill!(hk,UInt8(0));fill!(hd,UInt32(0));di=CuArray(hi);dk=CuArray(hk);dd=CuArray(hd);backend=KernelAbstractions.get_backend(di);blocks=cld(cld(f.n,f.base),256)
 search_kernel!(backend,256)(di,dk,dd,f.n,f.len,f.base,f.sd...;ndrange=blocks*256)
 CUDA.synchronize();copyto!(hi,di);copyto!(hk,dk);copyto!(hd,dd)
 for a in (di,dk,dd);CUDA.unsafe_free!(a);end
 nothing
end
function probe_launch!(w0,w1,lens,out,n)
 probe_kernel!(KernelAbstractions.get_backend(w0),256)(w0,w1,lens,out,n;ndrange=n)
end
study_main(search_call!,probe_launch!)
