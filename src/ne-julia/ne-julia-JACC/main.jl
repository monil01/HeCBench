using CUDA,JACC
include("../normal_device.jl")
const SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=1,shmem_size=0,sync=false)
function normal_kernel!(i,points,out,W,H)
 normal_site!(points,out,i,W,H);nothing
end
launch(c)=(JACC.parallel_for(normal_kernel!,SPEC,c.N,c.p,c.out,c.W,c.H);nothing)
include("../study_common.jl")
normal_study(launch)
