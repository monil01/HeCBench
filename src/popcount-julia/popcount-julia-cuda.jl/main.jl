using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function pc_kernel!(data,out,lut,n::Int32,which::Int32)
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    if i<=n
        @inbounds out[(which-Int32(1))*(n+Int32(2))+i+Int32(1)]=pc_value(data[i],which,lut)
    end
    return
end
function factory(data,out,lut,n)
    ()->nothing
    return which->(@cuda threads=256 blocks=cld(n,256) pc_kernel!(data,out,lut,Int32(n),which))
end
popcount_study(factory)
