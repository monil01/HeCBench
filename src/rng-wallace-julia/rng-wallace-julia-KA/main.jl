using CUDA,KernelAbstractions
const WALLACE_POOL_SIZE = 2048
const WALLACE_RUNS_PER_THREAD = 2
const WALLACE_NUM_THREADS = WALLACE_POOL_SIZE ÷ (4 * WALLACE_RUNS_PER_THREAD)
const WALLACE_MAX_OUTPUTS_PER_ITERATION = 4
const WALLACE_OUTPUT_COMBINE_COUNT = 1
const WALLACE_NUM_BLOCKS = 16384
const WALLACE_NUM_POOL_PASSES = 1
const WALLACE_NUM_OUTPUTS_PER_RUN = 1
const WALLACE_NUM_RANDOM_NUMBERS_PER_THREAD =
    WALLACE_NUM_OUTPUTS_PER_RUN * WALLACE_RUNS_PER_THREAD * WALLACE_MAX_OUTPUTS_PER_ITERATION
const WALLACE_TOTAL_NUM_THREADS = WALLACE_NUM_BLOCKS * WALLACE_NUM_THREADS
const WALLACE_TOTAL_POOL_SIZE = WALLACE_POOL_SIZE * WALLACE_NUM_BLOCKS
const WALLACE_NUM_RANDOM_NUMBERS_PER_BLOCK =
    WALLACE_NUM_RANDOM_NUMBERS_PER_THREAD * WALLACE_NUM_THREADS
const WALLACE_OUTPUT_SIZE = WALLACE_NUM_RANDOM_NUMBERS_PER_BLOCK * WALLACE_NUM_BLOCKS
const WALLACE_CHI2_COUNT = WALLACE_NUM_OUTPUTS_PER_RUN * WALLACE_NUM_BLOCKS
const WALLACE_CHI2_OFFSET = WALLACE_POOL_SIZE
const WALLACE_CHI2_SHARED_SIZE = 1

include(joinpath(@__DIR__,"..","wallace_math.jl"))
@kernel unsafe_indices=true function rng_wallace_kernel!(global_pool, generated, chi2, seed_in::UInt32)
    pool = @localmem(Float32, (2049,))
    tid = Int32(@index(Local,Linear)) - Int32(1)
    block = Int32(@index(Group,Linear)) - Int32(1)
    offset = Int32(WALLACE_POOL_SIZE) * block
    @inbounds for i in Int32(0):Int32(7)
        pool[Int(tid + Int32(WALLACE_NUM_THREADS) * i + Int32(1))] =
            global_pool[Int(offset + tid + Int32(WALLACE_NUM_THREADS) * i + Int32(1))]
    end
    @synchronize()

    m_seed = seed_in
    @inbounds for loop in UInt32(0):UInt32(WALLACE_NUM_OUTPUTS_PER_RUN - 1)
        m_seed = UInt32(1664525) * m_seed + UInt32(1013904223)
        intermediate = Int(loop) * 8 * WALLACE_TOTAL_NUM_THREADS +
            8 * WALLACE_NUM_THREADS * Int(block) + Int(tid)
        if tid == 0
            pool[WALLACE_CHI2_OFFSET + 1] =
                chi2[Int(block) * WALLACE_NUM_OUTPUTS_PER_RUN + Int(loop) + 1]
        end
        @synchronize()
        scale = pool[WALLACE_CHI2_OFFSET + 1]
        for i in 0:7
            generated[intermediate + i * WALLACE_NUM_THREADS + 1] =
                pool[i * WALLACE_NUM_THREADS + Int(tid) + 1] * scale
        end

        for _ in 1:WALLACE_NUM_POOL_PASSES
            lcg_a = UInt32(241)
            lcg_c = UInt32(59)
            mask = UInt32(255)
            s = (m_seed + UInt32(tid)) & mask
            s = (s * lcg_a + lcg_c) & mask
            r00 = pool[Int((s << UInt32(3)) + UInt32(1))]
            s = (s * lcg_a + lcg_c) & mask
            r10 = pool[Int((s << UInt32(3)) + UInt32(2))]
            s = (s * lcg_a + lcg_c) & mask
            r20 = pool[Int((s << UInt32(3)) + UInt32(3))]
            s = (s * lcg_a + lcg_c) & mask
            r30 = pool[Int((s << UInt32(3)) + UInt32(4))]
            s = (s * lcg_a + lcg_c) & mask
            r01 = pool[Int((s << UInt32(3)) + UInt32(5))]
            s = (s * lcg_a + lcg_c) & mask
            r11 = pool[Int((s << UInt32(3)) + UInt32(6))]
            s = (s * lcg_a + lcg_c) & mask
            r21 = pool[Int((s << UInt32(3)) + UInt32(7))]
            s = (s * lcg_a + lcg_c) & mask
            r31 = pool[Int((s << UInt32(3)) + UInt32(8))]

            @synchronize()
            r00, r10, r20, r30 = hadamard4x4a(r00, r10, r20, r30)
            pool[Int(tid + Int32(1))] = r00
            pool[Int(Int32(WALLACE_NUM_THREADS) + tid + Int32(1))] = r10
            pool[Int(Int32(2 * WALLACE_NUM_THREADS) + tid + Int32(1))] = r20
            pool[Int(Int32(3 * WALLACE_NUM_THREADS) + tid + Int32(1))] = r30

            r01, r11, r21, r31 = hadamard4x4b(r01, r11, r21, r31)
            pool[Int(Int32(4 * WALLACE_NUM_THREADS) + tid + Int32(1))] = r01
            pool[Int(Int32(5 * WALLACE_NUM_THREADS) + tid + Int32(1))] = r11
            pool[Int(Int32(6 * WALLACE_NUM_THREADS) + tid + Int32(1))] = r21
            pool[Int(Int32(7 * WALLACE_NUM_THREADS) + tid + Int32(1))] = r31
            @synchronize()
        end
    end

    @inbounds for i in Int32(0):Int32(7)
        global_pool[Int(offset + tid + Int32(WALLACE_NUM_THREADS) * i + Int32(1))] =
            pool[Int(tid + Int32(WALLACE_NUM_THREADS) * i + Int32(1))]
    end
end

const ST_KERNEL=rng_wallace_kernel!(CUDABackend(),256)
function study_launch!(d,seed)
 ST_KERNEL(d[1],d[2],d[3],seed;ndrange=257*256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("ka",study_launch!)
