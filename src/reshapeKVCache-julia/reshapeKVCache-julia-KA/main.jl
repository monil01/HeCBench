using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function cache_kernel(k,v,kc,vc,slots,H,D,B,x,ks,vs)
 token=@index(Group,Linear);tid=@index(Local,Linear)
 cache_lane(Int64(token-1),Int32(tid-1),k,v,kc,vc,slots,H,D,B,x,ks,vs)
end
cache_study((k,v,kc,vc,slots,H,D,B,x,ks,vs)->begin
 kernel=cache_kernel(CUDABackend(),512)
 ()->kernel(k,v,kc,vc,slots,H,D,B,x,ks,vs;ndrange=1024*512)
end)
