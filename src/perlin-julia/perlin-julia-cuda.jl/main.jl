using CUDA
include("../device_math.jl")
include("../study_common.jl")
function perlin_cuda(ystart,height,p,hash,gx,gy,out)
    px=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
    py=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y-Int32(1)+ystart
    pixel_noise!(px,py,ystart,height,p,hash,gx,gy,out)
    return
end
function observer_cuda(out,p,h,ns)
    threadIdx().x==1 && echo_noise!(out,p,h,ns)
    return
end
function launch_cuda(stream,ystart,height,p,hash,gx,gy,out)
    @cuda threads=(16,16) blocks=(3840,cld(height,16)) stream=stream perlin_cuda(ystart,height,p,hash,gx,gy,out)
end
observe_cuda(out,p,h,ns)=(@cuda threads=32 observer_cuda(out,p,h,ns);nothing)
perlin_study(CuArray,launch_cuda,observe_cuda)
