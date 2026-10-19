using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include("../study_device.jl")
function address_kernel!(i,rects,out,base,width,n)
 address_body!(rects,out,base,width,n,Int32(i)-Int32(1))
 nothing
end
function launch_address(a,b,p,w,n)
 spec=JACC.launch_spec(threads=256,blocks=cld(Int(n),256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(Int(n),256)*256,address_kernel!,a,b,p,w,n)
 nothing
end
include("../study_common.jl")
main()
