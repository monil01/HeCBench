# Sourcecopyright/terms in ../LICENSE.
using CUDA,JACC
include(joinpath(@__DIR__,"..","md5_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline search_callback!(i,fi,fk,fd,n,len,base,s0,s1,s2,s3)=study_search_thread!(i-1,fi,fk,fd,n,len,base,s0,s1,s2,s3)
@inline probe_callback!(i,w0,w1,lens,out,n)=study_probe_thread!(i,w0,w1,lens,out,n)
function search_call!(f,hi,hk,hd)
 fill!(hi,Int32(-1));fill!(hk,UInt8(0));fill!(hd,UInt32(0));di=CuArray(hi);dk=CuArray(hk);dd=CuArray(hd);backend=JACC.get_backend(:cuda);blocks=cld(cld(f.n,f.base),256)
 spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(search_callback!,spec,blocks*256,di,dk,dd,f.n,f.len,f.base,f.sd...)
 CUDA.synchronize();copyto!(hi,di);copyto!(hk,dk);copyto!(hd,dd)
 for a in (di,dk,dd);CUDA.unsafe_free!(a);end
 nothing
end
function probe_launch!(w0,w1,lens,out,n)
 backend=JACC.get_backend(:cuda)
 spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(probe_callback!,spec,n,w0,w1,lens,out,n)
end
study_main(search_call!,probe_launch!)
