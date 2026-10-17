using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function grrt_kernel!(o,v,task)
 x=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
 if x<8&&y<8;gr_pixel!(o,v,x,y,task);end
 return
end
function grrt_launch!(o,v,task)
 @cuda threads=(100,1) blocks=(1,8) grrt_kernel!(o,v,task)
end
if abspath(PROGRAM_FILE)==@__FILE__;study_main(grrt_launch!);end
