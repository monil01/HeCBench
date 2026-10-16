using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function channel_nhwc_ka!(n::Int32, cnum::Int32, hxw::Int32, x, sum, sumsq)
    c0 = Int32(@index(Group,Linear)) - Int32(1)
    tid = Int32(@index(Local,Linear))
    tid0 = tid - Int32(1)
    inner = n * hxw
    sval = Int32(0)
    qval = Int32(0)
    i0 = tid0
    while i0 < inner
        @inbounds v = x[i0 * cnum + c0 + Int32(1)]
        sval += v
        qval += v * v
        i0 += Int32(256)
    end
    sm1 = @localmem Int32 (256,)
    sm2 = @localmem Int32 (256,)
    sm1[tid] = sval
    sm2[tid] = qval
    @synchronize
    offset = Int32(256) ÷ Int32(2)
    while offset > 0
        if tid <= offset
            sm1[tid] += sm1[tid + offset]
            sm2[tid] += sm2[tid + offset]
        end
        @synchronize
        offset ÷= Int32(2)
    end
    if tid == Int32(1)
        @inbounds begin
            sum[c0 + Int32(1)] = sm1[Int32(1)]
            sumsq[c0 + Int32(1)] = sm2[Int32(1)]
        end
    end
end

@kernel function channel_nchw_ka!(n::Int32, cnum::Int32, hxw::Int32, x, sum, sumsq)
    c0 = Int32(@index(Group,Linear)) - Int32(1)
    tid = Int32(@index(Local,Linear))
    tid0 = tid - Int32(1)
    total = n * hxw
    sval = Int32(0)
    qval = Int32(0)
    i0 = tid0
    while i0 < total
        n0 = i0 ÷ hxw
        hw = i0 - n0 * hxw
        @inbounds v = x[(n0 * cnum + c0) * hxw + hw + Int32(1)]
        sval += v
        qval += v * v
        i0 += Int32(256)
    end
    sm1 = @localmem Int32 (256,)
    sm2 = @localmem Int32 (256,)
    sm1[tid] = sval
    sm2[tid] = qval
    @synchronize
    offset = Int32(256) ÷ Int32(2)
    while offset > 0
        if tid <= offset
            sm1[tid] += sm1[tid + offset]
            sm2[tid] += sm2[tid + offset]
        end
        @synchronize
        offset ÷= Int32(2)
    end
    if tid == Int32(1)
        @inbounds begin
            sum[c0 + Int32(1)] = sm1[Int32(1)]
            sumsq[c0 + Int32(1)] = sm2[Int32(1)]
        end
    end
end

function factory(input,sum,sumsq,nchw)
    kernel=nchw ? channel_nchw_ka!(get_backend(input),256) : channel_nhwc_ka!(get_backend(input),256)
    return ()->kernel(Int32(4),Int32(128),Int32(1024),input,sum,sumsq;ndrange=128*256)
end
channel_study(factory)
