using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function pc_ka!(data,out,lut,n::Int32,which::Int32)
    i=@index(Global,Linear)
    if i<=n
        @inbounds out[(which-Int32(1))*(n+Int32(2))+i+1]=pc_value(data[i],which,lut)
    end
end
function factory(data,out,lut,n)
    kernel=pc_ka!(CUDABackend(),256)
    return which->kernel(data,out,lut,Int32(n),which;ndrange=n)
end
popcount_study(factory)
