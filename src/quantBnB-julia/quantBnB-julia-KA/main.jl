using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function quantbnb_ka!(code, a, out, n::Int32)
    idx0 = Int32(@index(Global,Linear))-Int32(1)
    stride = Int32(262144)
    while idx0 < n
        @inbounds out[idx0 + Int32(1)] = dquantize(code, a[idx0 + Int32(1)])
        idx0 += stride
    end
end

function factory(code,input,output)
    kernel=quantbnb_ka!(get_backend(input),256)
    return ()->kernel(code,input,output,Int32(262144);ndrange=262144)
end
quantbnb_study(factory)
