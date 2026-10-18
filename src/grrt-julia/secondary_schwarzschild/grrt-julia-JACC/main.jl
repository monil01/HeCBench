using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const GRRT_BACKEND=JACC.get_backend(:cuda)
const GRRT_SPEC=JACC.LaunchSpec{typeof(GRRT_BACKEND)}(;threads=(100,1),blocks=(1,8),shmem_size=0,sync=false)
function grrt_jacc!(gx,gy,o,v,task)
 x=Int32(gx-1);y=Int32(gy-1)
 if x<8&&y<8;gr_pixel!(o,v,x,y,task);end
 nothing
end
function grrt_launch!(o,v,task)
 JACC.parallel_for(grrt_jacc!,GRRT_SPEC,(100,8),o,v,task)
end
study_main(grrt_launch!)
