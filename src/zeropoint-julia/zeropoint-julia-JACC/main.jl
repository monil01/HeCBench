using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function zero_point_jacc!(index,x_min, x_max, qmin::Int32, qmax::Int32, n::Int32,
                            preserve_sparsity::Bool, scale, zp)
    i = Int32(index)
    if i <= n
        @inbounds s, z = zero_point_one(x_min[i], x_max[i], qmin, qmax, preserve_sparsity)
        @inbounds scale[i] = s
        @inbounds zp[i] = z
    end
    return
end

function factory(lo,hi,s,z,n)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return ()->JACC.parallel_for(spec,Int(n),zero_point_jacc!,lo,hi,Int32(-127),Int32(127),n,true,s,z)
end
zeropoint_study(factory)
