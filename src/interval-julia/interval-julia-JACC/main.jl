using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include("../study_device.jl")
function interval_kernel!(i,out,counts,domain,choice)
 interval_body!(Int32(i)-Int32(1),out,counts,domain,choice);nothing
end
function echo_kernel!(i,out,domain)
 interval_echo_body!(Int32(i)-Int32(1),out,domain);nothing
end
function launch_main(s,choice)
 spec=JACC.launch_spec(threads=64,blocks=1024,shmem_size=0,sync=false)
 JACC.parallel_for(spec,65536,interval_kernel!,s.result.data,s.counts.data,s.domain,Int32(choice));nothing
end
function launch_echo(s)
 spec=JACC.launch_spec(threads=64,blocks=1024,shmem_size=0,sync=false)
 JACC.parallel_for(spec,65536,echo_kernel!,s.echo.data,s.domain);nothing
end
include("../study_common.jl")
main()
