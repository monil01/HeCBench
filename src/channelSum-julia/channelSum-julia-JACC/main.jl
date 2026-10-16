using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function channel_nhwc_jacc!(index,template,n::Int32, cnum::Int32, hxw::Int32, x, sum, sumsq)
    c0 = Int32(div(index-1,256)+1) - Int32(1)
    tid = Int32(mod(index-1,256)+1)
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
    scratch=JACC.shared(template)
    sm1=@view scratch[1:256]
    sm2=@view scratch[257:512]
    sm1[tid] = sval
    sm2[tid] = qval
    JACC.sync_workgroup()
    offset = Int32(256) ÷ Int32(2)
    while offset > 0
        if tid <= offset
            sm1[tid] += sm1[tid + offset]
            sm2[tid] += sm2[tid + offset]
        end
        JACC.sync_workgroup()
        offset ÷= Int32(2)
    end
    if tid == Int32(1)
        @inbounds begin
            sum[c0 + Int32(1)] = sm1[Int32(1)]
            sumsq[c0 + Int32(1)] = sm2[Int32(1)]
        end
    end
end

function channel_nchw_jacc!(index,template,n::Int32, cnum::Int32, hxw::Int32, x, sum, sumsq)
    c0 = Int32(div(index-1,256)+1) - Int32(1)
    tid = Int32(mod(index-1,256)+1)
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
    scratch=JACC.shared(template)
    sm1=@view scratch[1:256]
    sm2=@view scratch[257:512]
    sm1[tid] = sval
    sm2[tid] = qval
    JACC.sync_workgroup()
    offset = Int32(256) ÷ Int32(2)
    while offset > 0
        if tid <= offset
            sm1[tid] += sm1[tid + offset]
            sm2[tid] += sm2[tid + offset]
        end
        JACC.sync_workgroup()
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
    template=CUDA.zeros(Int32,512)
    spec=JACC.launch_spec(;threads=256,shmem_size=2048,sync=false)
    kernel=nchw ? channel_nchw_jacc! : channel_nhwc_jacc!
    return ()->JACC.parallel_for(spec,128*256,kernel,template,Int32(4),Int32(128),Int32(1024),input,sum,sumsq)
end
channel_study(factory)
