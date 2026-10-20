using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include("../study_device.jl")
function kernel_mt!(index,parameters,output,n)
 body_mt!(Int32(index)-Int32(1),parameters,output,n)
 nothing
end
function kernel_boxmuller!(index,output,n)
 body_boxmuller!(Int32(index)-Int32(1),output,n)
 nothing
end
function dispatch_mt!(p,r,n)
 spec=JACC.launch_spec(threads=128,blocks=32,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_mt!,p,r,n)
 nothing
end
function dispatch_boxmuller!(r,n)
 spec=JACC.launch_spec(threads=128,blocks=32,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_boxmuller!,r,n)
 nothing
end
include("../study_common.jl")
main()
