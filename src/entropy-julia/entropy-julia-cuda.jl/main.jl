using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function entropy_kernel!(out,input,table,w,h)
 x=(blockIdx().x-1)*blockDim().x+threadIdx().x
 y=(blockIdx().y-1)*blockDim().y+threadIdx().y
 counts=CUDA.@cuStaticSharedMem(Int32,4096)
 if x<=w && y<=h
  lane=(threadIdx().y-1)*16+threadIdx().x
  entropy_point!(out,input,table,counts,x,y,lane,w,h)
 end
 return
end
function entropy_launch!(o,a,t,w,h)
 @cuda threads=(16,16) blocks=(cld(w,16),cld(h,16)) entropy_kernel!(o,a,t,w,h)
end
if abspath(PROGRAM_FILE)==@__FILE__;study_main(entropy_launch!);end
