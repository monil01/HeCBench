using CUDA
include("../study_common.jl")
function cache_kernel(k,v,kc,vc,slots,H,D,B,x,ks,vs)
 cache_lane(Int64(blockIdx().x-1),Int32(threadIdx().x-1),k,v,kc,vc,slots,H,D,B,x,ks,vs);nothing
end
cache_study((k,v,kc,vc,slots,H,D,B,x,ks,vs)->()->(@cuda threads=512 blocks=1024 cache_kernel(k,v,kc,vc,slots,H,D,B,x,ks,vs)))
