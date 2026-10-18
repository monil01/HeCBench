using CUDA,KernelAbstractions
include("../study_common.jl")
const BLOCK_SIZE = Int32(256)
const BLOCK_SIZE_BITS = Int32(8)
const MERS_N = Int32(624)
const MERS_M = Int32(397)
const MERS_R = UInt32(31)
const MERS_A = UInt32(0x9908b0df)
const MAX_DEVICE_MEM = Int32(750_000_000)

@inline function place_load(g_s, block_word::Int32, byte_idx::Int32)
 @inbounds return Int32(reinterpret(Int8,g_s)[Int(Int32(4)*block_word+byte_idx+Int32(1))])
end
@inline function place_store!(g_s,block_word::Int32,byte_idx::Int32,val::Int32)
 @inbounds reinterpret(Int8,g_s)[Int(Int32(4)*block_word+byte_idx+Int32(1))]=val%Int8
 return
end

@inline function conflict_load(g_s, block_word::Int32, conflict_word::Int32, idx::Int32)
    return Int32(g_s[Int(block_word + conflict_word + idx + Int32(1))])
end

@inline function conflict_store!(g_s, block_word::Int32, conflict_word::Int32, idx::Int32, val::Int32)
    g_s[Int(block_word + conflict_word + idx + Int32(1))] = UInt32(val)
    return
end


@kernel function petrinet_kernel!(g_s,g_v,g_m,n::Int32,s::Int32,seed::Int32)
    tid0=Int32(@index(Local,Linear))-Int32(1);group0=Int32(@index(Group,Linear))-Int32(1)
    mt = @localmem UInt32 (624,)
    sums = @localmem Float32 (256,)
    maxs = @localmem UInt32 (256,)

    nsquare2 = n * n * Int32(2)
    conflict_word = nsquare2 >> Int32(2)
    words_per_block = conflict_word + nsquare2
    block_word = (group0) * words_per_block

    loop_num = nsquare2 >> (BLOCK_SIZE_BITS + Int32(2))
    @inbounds for i in Int32(0):(loop_num - Int32(1))
        g_s[Int(block_word + tid0 + (i << BLOCK_SIZE_BITS) + Int32(1))] = UInt32(0x01010101)
    end
    rem_words = conflict_word - (loop_num << BLOCK_SIZE_BITS)
    if tid0 < rem_words
        g_s[Int(block_word + tid0 + (loop_num << BLOCK_SIZE_BITS) + Int32(1))] = UInt32(0x01010101)
    end
    if tid0==Int32(0)
        mt[1] = UInt32(seed + group0)
        @inbounds for i in Int32(2):MERS_N
            prev = mt[Int(i - Int32(1))]
            mt[Int(i)] = UInt32(1812433253) * (prev ⊻ (prev >> UInt32(30))) + UInt32(i - Int32(1))
        end
    end
    @synchronize

    step = Int32(0)
    while step < s
    lower_mask = (UInt32(1) << MERS_R) - UInt32(1)
    upper_mask = UInt32(0xffffffff) << MERS_R
    span = MERS_N - MERS_M
    y = UInt32(0)

    if tid0 < span
        y = (mt[Int(tid0 + Int32(1))] & upper_mask) | (mt[Int(tid0 + Int32(2))] & lower_mask)
        y = mt[Int(tid0 + MERS_M + Int32(1))] ⊻ (y >> UInt32(1)) ⊻ ((y & UInt32(1)) != 0 ? MERS_A : UInt32(0))
    end
    @synchronize
    if tid0 < span
        mt[Int(tid0 + Int32(1))] = y
    end
    @synchronize

    thdx = tid0 + span
    if tid0 < span
        y = (mt[Int(thdx + Int32(1))] & upper_mask) | (mt[Int(thdx + Int32(2))] & lower_mask)
        y = mt[Int(tid0 + Int32(1))] ⊻ (y >> UInt32(1)) ⊻ ((y & UInt32(1)) != 0 ? MERS_A : UInt32(0))
    end
    @synchronize
    if tid0 < span
        mt[Int(thdx + Int32(1))] = y
    end
    @synchronize

    thdx += span
    if thdx < MERS_N - Int32(1)
        y = (mt[Int(thdx + Int32(1))] & upper_mask) | (mt[Int(thdx + Int32(2))] & lower_mask)
        y = mt[Int(tid0 + span + Int32(1))] ⊻ (y >> UInt32(1)) ⊻ ((y & UInt32(1)) != 0 ? MERS_A : UInt32(0))
    end
    @synchronize
    if thdx < MERS_N - Int32(1)
        mt[Int(thdx + Int32(1))] = y
    end
    @synchronize

    if tid0 == Int32(0)
        y = (mt[Int(MERS_N)] & upper_mask) | (mt[1] & lower_mask)
        mt[Int(MERS_N)] = mt[Int(MERS_M)] ⊻ (y >> UInt32(1)) ⊻ ((y & UInt32(1)) != 0 ? MERS_A : UInt32(0))
    end
    @synchronize

        for wave in Int32(0):Int32(2)
            thd_thrd=wave==Int32(2) ? Int32(112) : Int32(256)
            val=Int32(0)
            if tid0<thd_thrd;val=Int32(mt[Int(tid0+wave*Int32(256)+Int32(1))]%UInt32(nsquare2));end
            tr=val÷n;tc=val%n;eventstep=step+(wave==Int32(0) ? Int32(7) : wave==Int32(1) ? Int32(11) : Int32(13))
    val1 = tr == Int32(0) ? n + n - Int32(1) : tr - Int32(1)
    val2 = (tr & Int32(1)) != 0 ? (tc == n - Int32(1) ? Int32(0) : tc + Int32(1)) : tc
    val3 = tr == n + n - Int32(1) ? Int32(0) : tr + Int32(1)
    to_update = false
    mark1 = Int32(0)
    mark2 = Int32(0)

    if tid0 < thd_thrd
        mark1 = place_load(g_s, block_word, val1 * n + val2)
        mark2 = place_load(g_s, block_word, tr * n + tc)
        if mark1 > 0 && mark2 > 0
            to_update = true
            conflict_store!(g_s, block_word, conflict_word, tr * n + tc, eventstep)
        end
    end
    @synchronize

    if to_update
        to_update = ((eventstep & Int32(1)) == (tr & Int32(1))) ||
            (conflict_load(g_s, block_word, conflict_word, val1 * n + val2) != eventstep &&
             conflict_load(g_s, block_word, conflict_word, val3 * n + (val2 == Int32(0) ? n - Int32(1) : val2 - Int32(1))) != eventstep)
    end

    if to_update
        place_store!(g_s, block_word, val1 * n + val2, mark1 - Int32(1))
        place_store!(g_s, block_word, tr * n + tc, mark2 - Int32(1))
    end
    @synchronize
    if to_update
        place_store!(g_s, block_word, val3 * n + val2, place_load(g_s, block_word, val3 * n + val2) + Int32(1))
        right = tr * n + (tc == n - Int32(1) ? Int32(0) : tc + Int32(1))
        place_store!(g_s, block_word, right, place_load(g_s, block_word, right) + Int32(1))
    end
    @synchronize

        end
        step += MERS_N >> Int32(1)
    end

    sum = Float32(0)
    maxv = UInt32(0)
    @inbounds for i in Int32(0):(loop_num - Int32(1))
        data = g_s[Int(block_word + tid0 + (i << BLOCK_SIZE_BITS) + Int32(1))]
        t = data & UInt32(0xff); sum += Float32(t * t); maxv = ifelse(maxv < t, t, maxv)
        t = (data >> UInt32(8)) & UInt32(0xff); sum += Float32(t * t); maxv = ifelse(maxv < t, t, maxv)
        t = (data >> UInt32(16)) & UInt32(0xff); sum += Float32(t * t); maxv = ifelse(maxv < t, t, maxv)
        t = (data >> UInt32(24)) & UInt32(0xff); sum += Float32(t * t); maxv = ifelse(maxv < t, t, maxv)
    end
    i = conflict_word & Int32(0xff)
    base = loop_num * BLOCK_SIZE
    if tid0 <= i - Int32(1)
        data = g_s[Int(block_word + tid0 + base + Int32(1))]
        t = data & UInt32(0xff); sum += Float32(t * t); maxv = ifelse(maxv < t, t, maxv)
        t = (data >> UInt32(8)) & UInt32(0xff); sum += Float32(t * t); maxv = ifelse(maxv < t, t, maxv)
        t = (data >> UInt32(16)) & UInt32(0xff); sum += Float32(t * t); maxv = ifelse(maxv < t, t, maxv)
        t = (data >> UInt32(24)) & UInt32(0xff); sum += Float32(t * t); maxv = ifelse(maxv < t, t, maxv)
    end

    sums[Int(tid0 + Int32(1))] = sum
    maxs[Int(tid0 + Int32(1))] = maxv
    @synchronize

    stride = BLOCK_SIZE >> Int32(1)
    while stride > 0
        if tid0 < stride
            sums[Int(tid0 + Int32(1))] += sums[Int(tid0 + stride + Int32(1))]
            other = maxs[Int(tid0 + stride + Int32(1))]
            if maxs[Int(tid0 + Int32(1))] < other
                maxs[Int(tid0 + Int32(1))] = other
            end
        end
        @synchronize
        stride >>= Int32(1)
    end

    if tid0 == Int32(0)
        g_v[Int((group0+Int32(1)))] = sums[1] / Float32(nsquare2) - Float32(1)
        g_m[Int((group0+Int32(1)))] = Int32(maxs[1])
    end

end
function factory(g,v,m,n,s,seed)
 kernel=petrinet_kernel!(CUDABackend(),256)
 ()->kernel(g,v,m,n,s,seed;ndrange=1024)
end
main()=pns_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
