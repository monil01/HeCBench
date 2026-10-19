using CUDA
include(joinpath(@__DIR__,"..","device_math.jl"))
function tone_cuda!(input,output,av,gamma,c,delta,w,h)
    x=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    y=(blockIdx().y-1)*blockDim().y+threadIdx().y-1
    tone_pixel!(x,y,input,output,av,gamma,c,delta,w,h)
    nothing
end
function echo_cuda!(p,av,gamma,c,delta,w,h)
    if threadIdx().x==1;tone_echo!(p,av,gamma,c,delta,w,h);end
    nothing
end
launch!(i,o,av,gamma,c,delta,w,h)=@cuda threads=(16,16) blocks=(w÷16,h÷16) tone_cuda!(i,o,av,gamma,c,delta,w,h)
observe!(p,av,gamma,c,delta,w,h)=@cuda threads=32 blocks=1 echo_cuda!(p,av,gamma,c,delta,w,h)
include(joinpath(@__DIR__,"..","study_common.jl"))
tone_study(CuArray,launch!,observe!)
