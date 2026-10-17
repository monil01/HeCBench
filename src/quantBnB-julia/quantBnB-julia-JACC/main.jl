using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function quantbnb_jacc!(index,code, a, out, n::Int32)
    idx0 = Int32(index)-Int32(1)
    stride = Int32(262144)
    while idx0 < n
        @inbounds out[idx0 + Int32(1)] = dquantize(code, a[idx0 + Int32(1)])
        idx0 += stride
    end
    return
end

function factory(code,input,output)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return ()->JACC.parallel_for(spec,262144,quantbnb_jacc!,code,input,output,Int32(262144))
end
quantbnb_study(factory)
