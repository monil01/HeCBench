using CUDA
using Printf

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

mutable struct KissState
    z::UInt32
    w::UInt32
    jsr::UInt32
    jcong::UInt32
    cached::Bool
    cn::Float64
end

KissState() = KissState(0x159a55e5, 0x1f123bb5, 0x075bcd15, 0x16a474c0, false, 0.0)

function kiss!(s::KissState)
    s.z = UInt32(36969) * (s.z & UInt32(0xffff)) + (s.z >> UInt32(16))
    s.w = UInt32(18000) * (s.w & UInt32(0xffff)) + (s.w >> UInt32(16))
    mwc = (s.z << UInt32(16)) + s.w
    s.jsr ⊻= s.jsr << UInt32(17)
    s.jsr ⊻= s.jsr >> UInt32(13)
    s.jsr ⊻= s.jsr << UInt32(5)
    s.jcong = UInt32(69069) * s.jcong + UInt32(1234567)
    return (mwc ⊻ s.jcong) + s.jsr
end

function rand_uniform!(s::KissState)
    x = UInt64(kiss!(s))
    x = (x << UInt32(32)) | UInt64(kiss!(s))
    return Float64(x) * 5.4210108624275221703311375920553e-20
end

function randn_box_muller!(s::KissState)
    if s.cached
        s.cached = false
        return s.cn
    end
    a = sqrt(-2.0 * log(rand_uniform!(s)))
    b = 6.283185307179586476925286766559 * rand_uniform!(s)
    s.cn = sin(b) * a
    s.cached = true
    return cos(b) * a
end

function make_chi2_scale!(s::KissState, n::UInt32)
    chic1 = sqrt(sqrt(1.0 - 1.0 / Float64(n)))
    chic2 = sqrt(1.0 - chic1 * chic1)
    return Float32(chic1 + chic2 * randn_box_muller!(s))
end

@inline function hadamard4x4a(p::Float32, q::Float32, r::Float32, s::Float32)
    t = (p + q + r + s) / Float32(2)
    return p - t, q - t, t - r, t - s
end

@inline function hadamard4x4b(p::Float32, q::Float32, r::Float32, s::Float32)
    t = (p + q + r + s) / Float32(2)
    return t - p, t - q, r - t, s - t
end

function rng_wallace_kernel!(global_pool, generated, chi2, seed_in::UInt32)
    pool = @cuStaticSharedMem(Float32, 2049)
    tid = threadIdx().x - Int32(1)
    block = blockIdx().x - Int32(1)
    offset = Int32(WALLACE_POOL_SIZE) * block
    @inbounds for i in Int32(0):Int32(7)
        pool[Int(tid + Int32(WALLACE_NUM_THREADS) * i + Int32(1))] =
            global_pool[Int(offset + tid + Int32(WALLACE_NUM_THREADS) * i + Int32(1))]
    end
    sync_threads()

    m_seed = seed_in
    @inbounds for loop in UInt32(0):UInt32(WALLACE_NUM_OUTPUTS_PER_RUN - 1)
        m_seed = UInt32(1664525) * m_seed + UInt32(1013904223)
        intermediate = Int(loop) * 8 * WALLACE_TOTAL_NUM_THREADS +
            8 * WALLACE_NUM_THREADS * Int(block) + Int(tid)
        if tid == 0
            pool[WALLACE_CHI2_OFFSET + 1] =
                chi2[Int(block) * WALLACE_NUM_OUTPUTS_PER_RUN + Int(loop) + 1]
        end
        sync_threads()
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

            sync_threads()
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
            sync_threads()
        end
    end

    @inbounds for i in Int32(0):Int32(7)
        global_pool[Int(offset + tid + Int32(WALLACE_NUM_THREADS) * i + Int32(1))] =
            pool[Int(tid + Int32(WALLACE_NUM_THREADS) * i + Int32(1))]
    end
    return
end

function reference!(seed_in::UInt32, pool_global, generated, chi2, num_blocks::Int)
    lcg_a = UInt32(241)
    lcg_c = UInt32(59)
    mask = UInt32(255)
    pool = Vector{Float32}(undef, WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE)
    for block in 0:(num_blocks - 1)
        t_seed = seed_in
        offset = WALLACE_POOL_SIZE * block
        @inbounds for tid in 0:(WALLACE_NUM_THREADS - 1), i in 0:7
            pool[tid + WALLACE_NUM_THREADS * i + 1] =
                pool_global[offset + tid + WALLACE_NUM_THREADS * i + 1]
        end
        @inbounds for loop in 0:(WALLACE_NUM_OUTPUTS_PER_RUN - 1)
            t_seed = UInt32(1664525) * t_seed + UInt32(1013904223)
            pool[WALLACE_CHI2_OFFSET + 1] = chi2[block * WALLACE_NUM_OUTPUTS_PER_RUN + loop + 1]
            scale = pool[WALLACE_CHI2_OFFSET + 1]
            for tid in 0:(WALLACE_NUM_THREADS - 1)
                intermediate = loop * 8 * WALLACE_TOTAL_NUM_THREADS +
                    8 * WALLACE_NUM_THREADS * block + tid
                for i in 0:7
                    generated[intermediate + i * WALLACE_NUM_THREADS + 1] =
                        pool[i * WALLACE_NUM_THREADS + tid + 1] * scale
                end
            end
            for _ in 1:WALLACE_NUM_POOL_PASSES
                rin = Matrix{Float32}(undef, WALLACE_NUM_THREADS, 8)
                for tid in 0:(WALLACE_NUM_THREADS - 1)
                    s = (t_seed + UInt32(tid)) & mask
                    for lane in 0:7
                        s = (s * lcg_a + lcg_c) & mask
                        rin[tid + 1, lane + 1] = pool[Int((s << UInt32(3)) + UInt32(lane) + UInt32(1))]
                    end
                    rin[tid + 1, 1], rin[tid + 1, 2], rin[tid + 1, 3], rin[tid + 1, 4] =
                        hadamard4x4a(rin[tid + 1, 1], rin[tid + 1, 2], rin[tid + 1, 3], rin[tid + 1, 4])
                end
                for tid in 0:(WALLACE_NUM_THREADS - 1)
                    pool[tid + 1] = rin[tid + 1, 1]
                    pool[WALLACE_NUM_THREADS + tid + 1] = rin[tid + 1, 2]
                    pool[2 * WALLACE_NUM_THREADS + tid + 1] = rin[tid + 1, 3]
                    pool[3 * WALLACE_NUM_THREADS + tid + 1] = rin[tid + 1, 4]
                    rin[tid + 1, 5], rin[tid + 1, 6], rin[tid + 1, 7], rin[tid + 1, 8] =
                        hadamard4x4b(rin[tid + 1, 5], rin[tid + 1, 6], rin[tid + 1, 7], rin[tid + 1, 8])
                    pool[4 * WALLACE_NUM_THREADS + tid + 1] = rin[tid + 1, 5]
                    pool[5 * WALLACE_NUM_THREADS + tid + 1] = rin[tid + 1, 6]
                    pool[6 * WALLACE_NUM_THREADS + tid + 1] = rin[tid + 1, 7]
                    pool[7 * WALLACE_NUM_THREADS + tid + 1] = rin[tid + 1, 8]
                end
            end
        end
        @inbounds for tid in 0:(WALLACE_NUM_THREADS - 1), i in 0:7
            pool_global[offset + tid + WALLACE_NUM_THREADS * i + 1] =
                pool[tid + WALLACE_NUM_THREADS * i + 1]
        end
    end
    return
end

function main()
    length(ARGS) == 1 || error("Usage: main.jl <repeat>")
    repeat = parse(Int, ARGS[1])

    state = KissState()
    host_pool = Vector{Float32}(undef, WALLACE_TOTAL_POOL_SIZE)
    pool_ref = similar(host_pool)
    @inbounds for i in eachindex(host_pool)
        x = Float32(randn_box_muller!(state))
        host_pool[i] = x
        pool_ref[i] = x
    end
    chi2 = Vector{Float32}(undef, WALLACE_CHI2_COUNT)
    @inbounds for i in eachindex(chi2)
        chi2[i] = make_chi2_scale!(state, UInt32(WALLACE_TOTAL_POOL_SIZE))
    end
    random_numbers_ref = Vector{Float32}(undef, WALLACE_OUTPUT_SIZE)

    d_pool = CuArray(host_pool)
    d_chi2 = CuArray(chi2)
    d_random = CuArray{Float32}(undef, WALLACE_OUTPUT_SIZE)
    seed = UInt32(1)
    grid = WALLACE_NUM_BLOCKS
    threads = WALLACE_NUM_THREADS

    for _ in 1:30
        @cuda threads=threads blocks=grid rng_wallace_kernel!(d_pool, d_random, d_chi2, seed)
    end
    for _ in 1:30
        reference!(seed, pool_ref, random_numbers_ref, chi2, WALLACE_NUM_BLOCKS)
    end

    random_numbers = Array(d_random)
    host_pool = Array(d_pool)
    ok = true
    @inbounds for i in eachindex(random_numbers)
        if abs(random_numbers_ref[i] - random_numbers[i]) > 1.0f-3
            @printf("randNumbers mismatch at index %d: %f %f\n", i - 1, random_numbers_ref[i], random_numbers[i])
            ok = false
            break
        end
    end
    if ok
        @inbounds for i in eachindex(host_pool)
            if abs(pool_ref[i] - host_pool[i]) > 1.0f-3
                @printf("Pool mismatch at index %d: %f %f\n", i - 1, pool_ref[i], host_pool[i])
                ok = false
                break
            end
        end
    end
    println(ok ? "PASS" : "FAIL")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=grid rng_wallace_kernel!(d_pool, d_random, d_chi2, seed)
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (us)\n", (time_ns() - t0) * 1e-3 / repeat)
end

main()
