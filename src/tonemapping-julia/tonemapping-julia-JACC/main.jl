using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","device_math.jl"))
function tone_jacc!(i,input,output,av,gamma,c,delta,w,h)
    x,y=tone_coords(i,w)
    tone_pixel!(x,y,input,output,av,gamma,c,delta,w,h)
    nothing
end
function echo_jacc!(i,p,av,gamma,c,delta,w,h)
    if i==1;tone_echo!(p,av,gamma,c,delta,w,h);end
    nothing
end
launch!(i,o,av,gamma,c,delta,w,h)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=Int(w*h)÷256,shmem_size=0,sync=false),Int(w*h),tone_jacc!,i,o,av,gamma,c,delta,w,h)
observe!(p,av,gamma,c,delta,w,h)=JACC.parallel_for(JACC.launch_spec(threads=32,blocks=1,shmem_size=0,sync=false),1,echo_jacc!,p,av,gamma,c,delta,w,h)
include(joinpath(@__DIR__,"..","study_common.jl"))
tone_study(JACC.array,launch!,observe!)
