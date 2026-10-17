include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function search_call!(f,hi,hk,hd)
 fill!(hi,Int32(-1));fill!(hk,UInt8(0));fill!(hd,UInt32(0));di=CuArray(hi);dk=CuArray(hk);dd=CuArray(hd)
 @cuda threads=256 blocks=cld(cld(f.n,f.base),256) search_kernel!(di,dk,dd,f.n,f.len,f.base,f.sd...)
 CUDA.synchronize();copyto!(hi,di);copyto!(hk,dk);copyto!(hd,dd)
 for a in (di,dk,dd);CUDA.unsafe_free!(a);end
 nothing
end
function probe_launch!(w0,w1,lens,out,n)
 @cuda threads=256 blocks=cld(n,256) probe_kernel!(w0,w1,lens,out,n)
end
study_main(search_call!,probe_launch!)
