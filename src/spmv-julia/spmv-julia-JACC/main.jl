using CUDA
import JACC
JACC.@init_backend
include("../study_device.jl")
function dense_jacc!(i,a,x,y,n)
 dense_body!(a,x,y,n,Int(i))
 return nothing
end
function launch(a,x,y,n)
 physical=cld(n,256)*256
 spec=JACC.launch_spec(;threads=256,blocks=physical÷256,shmem_size=0,sync=false)
 JACC.parallel_for(spec,physical,dense_jacc!,a,x,y,n)
end
include("../study_common.jl")
run_study(launch)
