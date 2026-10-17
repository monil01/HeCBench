using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function scaled_fp4_ka!(input,output,n)
    i=@index(Global,Linear);lid=@index(Local,Linear)
    tile=@localmem Float16 (128,)
    @inbounds tile[lid]=input[i]
    @synchronize
    start=div(lid-1,32)*32;maxval=Float16(0)
    for lane in 1:32;maxval=max(maxval,abs(@inbounds tile[start+lane]));end
    @inbounds output[i]=scaled_fp4(tile[lid],maxval)
end
function factory(input,output,n)
    kernel=scaled_fp4_ka!(get_backend(input),128)
    return ()->kernel(input,output,n;ndrange=n)
end
mxfp4_study(factory)
