using CUDA,JACC
include("../arithmetic_device.jl")
function arithmetic_kernel!(i,data,nIters,v1,v2,op,L)
 arithmetic_site!(data,i,nIters,v1,v2,op,L);nothing
end
function launch(c)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=c.N÷256,shmem_size=0,sync=false)
 JACC.parallel_for(arithmetic_kernel!,spec,c.N,c.data,c.R,c.v1,c.v2,Val(c.op),Val(c.L));nothing
end
include("../study_common.jl")
arithmetic_study(launch)
