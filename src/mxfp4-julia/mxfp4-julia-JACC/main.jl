using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function scaled_fp4_jacc!(i,input,output)
    lid=mod(i-1,128)+1;blockstart=div(i-1,128)*128
    tile=JACC.shared(@view input[blockstart+1:blockstart+128])
    start=div(lid-1,32)*32;maxval=Float16(0)
    for lane in 1:32;maxval=max(maxval,abs(@inbounds tile[start+lane]));end
    @inbounds output[i]=scaled_fp4(tile[lid],maxval)
end
function factory(input,output,n)
    spec=JACC.launch_spec(;threads=128,shmem_size=256,sync=false)
    return ()->JACC.parallel_for(spec,n,scaled_fp4_jacc!,input,output)
end
mxfp4_study(factory)
