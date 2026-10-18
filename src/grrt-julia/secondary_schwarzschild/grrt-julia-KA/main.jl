using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function grrt_ka!(o,v,task)
 gx,gy=@index(Global,NTuple);x=Int32(gx-1);y=Int32(gy-1)
 if x<8&&y<8;gr_pixel!(o,v,x,y,task);end
end
function grrt_launch!(o,v,task)
 grrt_ka!(CUDABackend(),(100,1))(o,v,task;ndrange=(100,8))
end
study_main(grrt_launch!)
