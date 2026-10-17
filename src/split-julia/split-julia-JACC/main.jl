using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_split!(i,keys_in,keys_out,scratch)
    j=Int32(i)-Int32(1);bid=j÷Int32(128)+Int32(1);tid=j%Int32(128)+Int32(1);base=(bid-Int32(1))*Int32(512);sh=JACC.shared(scratch)
    i0 = (tid - Int32(1)) * Int32(4)
    @inbounds begin
        sh[i0 + Int32(1)] = keys_in[base + i0 + Int32(1)]
        sh[i0 + Int32(2)] = keys_in[base + i0 + Int32(2)]
        sh[i0 + Int32(3)] = keys_in[base + i0 + Int32(3)]
        sh[i0 + Int32(4)] = keys_in[base + i0 + Int32(4)]
    end
    JACC.sync_workgroup()

    if tid == Int32(1)
        i = Int32(1)
        while i <= Int32(16)
            @inbounds sh[Int32(512) + i] = UInt32(0)
            i += Int32(1)
        end

        i = Int32(1)
        while i <= Int32(512)
            v = @inbounds sh[i]
            count_idx = Int32(512) + Int32(v) + Int32(1)
            @inbounds sh[count_idx] += UInt32(1)
            i += Int32(1)
        end

        pos = Int32(1)
        bucket = UInt32(0)
        while bucket <= UInt32(15)
            c = @inbounds sh[Int32(512) + Int32(bucket) + Int32(1)]
            j = UInt32(0)
            while j < c
                @inbounds sh[pos] = bucket
                pos += Int32(1)
                j += UInt32(1)
            end
            bucket += UInt32(1)
        end
    end
    JACC.sync_workgroup()

    @inbounds begin
        keys_out[base + i0 + Int32(1)] = sh[i0 + Int32(1)]
        keys_out[base + i0 + Int32(2)] = sh[i0 + Int32(2)]
        keys_out[base + i0 + Int32(3)] = sh[i0 + Int32(3)]
        keys_out[base + i0 + Int32(4)] = sh[i0 + Int32(4)]
    end
    nothing
end
function study_prepare(src,dst,n)
    scratch=CUDA.zeros(UInt32,528);backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=128,blocks=n÷512,shmem_size=528*4,sync=false)
    ()->JACC.parallel_for(portable_split!,spec,n÷4,src,dst,scratch)
end
exit(study_main(study_prepare))
