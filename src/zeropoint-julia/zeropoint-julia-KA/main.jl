using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function zero_point_ka!(x_min, x_max, qmin::Int32, qmax::Int32, n::Int32,
                            preserve_sparsity::Bool, scale, zp)
    i = Int32(@index(Global,Linear))
    if i <= n
        @inbounds s, z = zero_point_one(x_min[i], x_max[i], qmin, qmax, preserve_sparsity)
        @inbounds scale[i] = s
        @inbounds zp[i] = z
    end
end

function factory(lo,hi,s,z,n)
    kernel=zero_point_ka!(get_backend(lo),256)
    return ()->kernel(lo,hi,Int32(-127),Int32(127),n,true,s,z;ndrange=Int(n))
end
zeropoint_study(factory)
