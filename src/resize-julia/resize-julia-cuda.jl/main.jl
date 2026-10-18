using CUDA
include("../resize_device.jl")
function resize_kernel!(mode,out,inp,outsize,oh,ow,ih,iw,fy,fx)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 resize_site!(mode,out,inp,i,blockDim().x*gridDim().x,outsize,oh,ow,ih,iw,fy,fx)
 nothing
end
launch(c)=(@cuda threads=256 blocks=29184 resize_kernel!(c.mode,c.output,c.input,UInt64(c.no),c.oh,c.ow,c.ih,c.iw,c.fy,c.fx);nothing)
include("../study_common.jl")
resize_study(launch)
