using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","device_math.jl"))
@kernel function tone_ka!(input,output,av,gamma,c,delta,w,h)
    i=@index(Global,Linear)
    x,y=tone_coords(i,w)
    tone_pixel!(x,y,input,output,av,gamma,c,delta,w,h)
    nothing
end
@kernel function echo_ka!(p,av,gamma,c,delta,w,h)
    i=@index(Global,Linear)
    if i==1;tone_echo!(p,av,gamma,c,delta,w,h);end
    nothing
end
function ka_array(v)
    a=KernelAbstractions.allocate(CUDA.CUDABackend(),eltype(v),length(v))
    copyto!(a,v)
    a
end
launch!(i,o,av,gamma,c,delta,w,h)=tone_ka!(CUDA.CUDABackend(),256)(i,o,av,gamma,c,delta,w,h;ndrange=Int(w*h))
observe!(p,av,gamma,c,delta,w,h)=echo_ka!(CUDA.CUDABackend(),32)(p,av,gamma,c,delta,w,h;ndrange=1)
include(joinpath(@__DIR__,"..","study_common.jl"))
tone_study(ka_array,launch!,observe!)
