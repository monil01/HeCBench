using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function cache_kernel(i,k,v,kc,vc,slots,H,D,B,x,ks,vs)
 lane=i-1;cache_lane(Int64(lane÷512),Int32(lane%512),k,v,kc,vc,slots,H,D,B,x,ks,vs);nothing
end
cache_study((k,v,kc,vc,slots,H,D,B,x,ks,vs)->begin
 spec=JACC.launch_spec(;threads=512,blocks=1024,shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,1024*512,cache_kernel,k,v,kc,vc,slots,H,D,B,x,ks,vs)
end)
