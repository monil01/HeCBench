using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function ao_kernel!(x,y,img)
 render_lane!(Int32(x-1),Int32(y-1),img)
end
function factory(img)
 spec=JACC.launch_spec(;threads=(16,16),blocks=(16,16),shmem_size=0,sync=false);()->JACC.parallel_for(spec,(256,256),ao_kernel!,img)
end
ao_study(factory)
