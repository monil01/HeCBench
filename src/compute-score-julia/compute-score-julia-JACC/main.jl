using CUDA,JACC
const MANUAL_VECTOR=8
const NUM_THREADS_PER_WG=64
const BLOOM_1=5
const BLOOM_2=UInt32(0x7ffff)
const DOC_ENDING_TAG=UInt32(0xffffffff)
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function mulfp(weight::UInt64, freq::UInt32)
    part1 = UInt32(weight & UInt64(0xfffff))
    part2 = UInt32((weight >> 24) & UInt64(0xffff))
    res1 = part1 * freq
    res2 = part2 * freq
    return UInt64(res1) + (UInt64(res2) << 24)
end

function compute_kernel!(index,template,d1, d2, weights1, weights2, bloom, partial_hi, partial_lo)
    partial = JACC.shared(template)
    gid0 = Int32(index)-Int32(1)
    sum = UInt64(0)
    base = gid0 * Int32(MANUAL_VECTOR)

    @inbounds for j in Int32(0):(Int32(MANUAL_VECTOR) - Int32(1))
        curr = d1[base + j + Int32(1)]
        if curr != DOC_ENDING_TAG
            freq = curr & UInt32(0xff)
            word_id = curr >> 8
            hash1 = word_id >> BLOOM_1
            hash2 = word_id & BLOOM_2
            inh1 = ((bloom[Int32(hash1 >> 5) + Int32(1)] >> (hash1 & UInt32(0x1f))) & UInt32(1)) != 0
            inh2 = ((bloom[Int32(hash2 >> 5) + Int32(1)] >> (hash2 & UInt32(0x1f))) & UInt32(1)) != 0
            if inh1 && inh2
                sum += mulfp(weights1[Int32(word_id) + Int32(1)], freq)
            end
        end
    end

    @inbounds for j in Int32(0):(Int32(MANUAL_VECTOR) - Int32(1))
        curr = d2[base + j + Int32(1)]
        if curr != DOC_ENDING_TAG
            freq = curr & UInt32(0xff)
            word_id = curr >> 8
            hash1 = word_id >> BLOOM_1
            hash2 = word_id & BLOOM_2
            inh1 = ((bloom[Int32(hash1 >> 5) + Int32(1)] >> (hash1 & UInt32(0x1f))) & UInt32(1)) != 0
            inh2 = ((bloom[Int32(hash2 >> 5) + Int32(1)] >> (hash2 & UInt32(0x1f))) & UInt32(1)) != 0
            if inh1 && inh2
                sum += mulfp(weights2[Int32(word_id) + Int32(1)], freq)
            end
        end
    end

    @inbounds partial[Int32(mod(index-1,8)+1)] = sum
    JACC.sync_workgroup()
    if Int32(mod(index-1,8)+1) == Int32(1)
        final = UInt64(0)
        @inbounds for i in Int32(1):Int32(NUM_THREADS_PER_WG ÷ MANUAL_VECTOR)
            final += partial[i]
        end
        @inbounds partial_hi[Int32(div(index-1,8)+1)] = UInt32(final >> 32)
        @inbounds partial_lo[Int32(div(index-1,8)+1)] = UInt32(final & UInt64(0xffffffff))
    end
    return
end

function reduction_kernel!(index,doc_info, partial_hi, partial_lo, result)
    gid0 = Int32(index)-Int32(1)
    info = @inbounds doc_info[gid0 + Int32(1)]
    start = UInt32(info >> 32)
    stop = UInt32(info & UInt64(0xffffffff))
    total = UInt64(0)
    i = start
    @inbounds while i <= stop
        upper = UInt64(partial_hi[Int32(i) + Int32(1)])
        lower = UInt64(partial_lo[Int32(i) + Int32(1)])
        total += (upper << 32) | lower
        i += UInt32(1)
    end
    @inbounds result[gid0 + Int32(1)] = total
    return
end

function factory(a,z,w1,w2,bh,hi,lo,info,out,lines)
 template=CUDA.zeros(UInt64,8)
 spec=JACC.launch_spec(;threads=8,shmem_size=64,sync=false)
 red=JACC.launch_spec(;threads=64,shmem_size=0,sync=false)
 return ()->begin
 JACC.parallel_for(spec,lines*8,compute_kernel!,template,a,z,w1,w2,bh,hi,lo)
 JACC.parallel_for(red,64,reduction_kernel!,info,hi,lo,out)
 end
end
score_study(factory)
