using CUDA
using Printf

const FLOAT = Float32
const NTHR_PER_BLK = Int32(256)
const NBLOCK = Int32(56 * 4)
const NPOINT = Int32(NBLOCK * NTHR_PER_BLK)
const NEQ = Int32(100000)
const NGEN_PER_BLOCK = Int32(5000)
const DELTA = FLOAT(2.0)
const FOUR = FLOAT(4.0)
const TWO = FLOAT(2.0)
const ONE = FLOAT(1.0)
const HALF = FLOAT(0.5)
const ZERO = FLOAT(0.0)
const LCG_M = UInt32(2147483648)
const LCG_A = UInt32(26757677)
const LCG_C = UInt32(1)

@inline function lcg_random(seed::UInt32)
    next = (LCG_A * seed + LCG_C) % LCG_M
    return next, FLOAT(next) / FLOAT(LCG_M)
end

@inline function wave_function(x1, y1, z1, x2, y2, z2)
    r1 = sqrt(x1*x1 + y1*y1 + z1*z1)
    r2 = sqrt(x2*x2 + y2*y2 + z2*z2)
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    r12 = sqrt(dx*dx + dy*dy + dz*dz)
    return (ONE + HALF * r12) * exp(-TWO * (r1 + r2))
end

function initran_kernel!(seed::UInt32, states)
    i0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    idx = i0 + Int32(1)
    s = seed ⊻ UInt32(i0)
    s, _ = lcg_random(s)
    states[idx] = s
    return
end

function initialize_kernel!(x1, y1, z1, x2, y2, z2, psi, states)
    i0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    idx = i0 + Int32(1)
    s = states[idx]
    s, r = lcg_random(s); x1[idx] = (r - HALF) * FOUR
    s, r = lcg_random(s); y1[idx] = (r - HALF) * FOUR
    s, r = lcg_random(s); z1[idx] = (r - HALF) * FOUR
    s, r = lcg_random(s); x2[idx] = (r - HALF) * FOUR
    s, r = lcg_random(s); y2[idx] = (r - HALF) * FOUR
    s, r = lcg_random(s); z2[idx] = (r - HALF) * FOUR
    psi[idx] = wave_function(x1[idx], y1[idx], z1[idx], x2[idx], y2[idx], z2[idx])
    states[idx] = s
    return
end

function zero_stats_kernel!(stats)
    i0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    idx = i0 + Int32(1)
    n = Int(NPOINT)
    stats[idx] = ZERO
    stats[n + Int(idx)] = ZERO
    stats[2n + Int(idx)] = ZERO
    stats[3n + Int(idx)] = ZERO
    return
end

function propagate_kernel!(nstep::Int32, x1, y1, z1, x2, y2, z2, psi, stats, states)
    i0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    idx = i0 + Int32(1)
    xi1 = x1[idx]; yi1 = y1[idx]; zi1 = z1[idx]
    xi2 = x2[idx]; yi2 = y2[idx]; zi2 = z2[idx]
    p = psi[idx]
    s = states[idx]
    n = Int(NPOINT)

    for _ in Int32(1):nstep
        s, r = lcg_random(s); x1new = xi1 + (r - HALF) * DELTA
        s, r = lcg_random(s); y1new = yi1 + (r - HALF) * DELTA
        s, r = lcg_random(s); z1new = zi1 + (r - HALF) * DELTA
        s, r = lcg_random(s); x2new = xi2 + (r - HALF) * DELTA
        s, r = lcg_random(s); y2new = yi2 + (r - HALF) * DELTA
        s, r = lcg_random(s); z2new = zi2 + (r - HALF) * DELTA
        pnew = wave_function(x1new, y1new, z1new, x2new, y2new, z2new)
        s, r = lcg_random(s)
        if pnew * pnew > p * p * r
            stats[3n + Int(idx)] += ONE
            p = pnew
            xi1 = x1new; yi1 = y1new; zi1 = z1new
            xi2 = x2new; yi2 = y2new; zi2 = z2new
        end
        r1 = sqrt(xi1*xi1 + yi1*yi1 + zi1*zi1)
        r2 = sqrt(xi2*xi2 + yi2*yi2 + zi2*zi2)
        dx = xi1 - xi2; dy = yi1 - yi2; dz = zi1 - zi2
        r12 = sqrt(dx*dx + dy*dy + dz*dz)
        stats[Int(idx)] += r1
        stats[n + Int(idx)] += r2
        stats[2n + Int(idx)] += r12
    end

    x1[idx] = xi1; y1[idx] = yi1; z1[idx] = zi1
    x2[idx] = xi2; y2[idx] = yi2; z2[idx] = zi2
    psi[idx] = p
    states[idx] = s
    return
end

function sum_within_blocks_kernel!(n::Int32, data, offset::Int32, blocksums)
    tid0 = Int32(threadIdx().x - Int32(1))
    bid0 = Int32(blockIdx().x - Int32(1))
    stride = Int32(blockDim().x * gridDim().x)
    i0 = Int32(blockDim().x * bid0 + tid0)
    shared = @cuStaticSharedMem(FLOAT, 512)
    total = ZERO
    while i0 < n
        total += data[Int(offset + i0) + 1]
        i0 += stride
    end
    shared[threadIdx().x] = total
    sync_threads()
    step = Int32(128)
    while step > 0
        if tid0 < step && tid0 + step < blockDim().x
            shared[threadIdx().x] += shared[threadIdx().x + step]
        end
        sync_threads()
        step >>= 1
    end
    if tid0 == 0
        blocksums[blockIdx().x] = shared[1]
    end
    return
end

function reduce_stats!(stats, statsum, blocksums)
    for what0 in Int32(0):Int32(3)
        @cuda threads=Int(NTHR_PER_BLK) blocks=Int(NBLOCK) sum_within_blocks_kernel!(
            NPOINT, stats, what0 * NPOINT, blocksums)
        @cuda threads=Int(NBLOCK) blocks=1 sum_within_blocks_kernel!(
            NBLOCK, blocksums, Int32(0), view(statsum, Int(what0) + 1:Int(what0) + 1))
    end
end

function main()
    if isempty(ARGS)
        println("Usage: main.jl <number of blocks to sample>")
        exit(1)
    end
    nsample = parse(Int, ARGS[1])
    check = "--check" in ARGS[2:end]

    x1 = CUDA.zeros(FLOAT, Int(NPOINT)); y1 = similar(x1); z1 = similar(x1)
    x2 = similar(x1); y2 = similar(x1); z2 = similar(x1)
    psi = similar(x1)
    stats = CUDA.zeros(FLOAT, 4 * Int(NPOINT))
    statsum = CUDA.zeros(FLOAT, 4)
    blocksums = CUDA.zeros(FLOAT, Int(NBLOCK))
    states = CUDA.zeros(UInt32, Int(NPOINT))

    @cuda threads=Int(NTHR_PER_BLK) blocks=Int(NBLOCK) initran_kernel!(UInt32(5551212), states)
    @cuda threads=Int(NTHR_PER_BLK) blocks=Int(NBLOCK) initialize_kernel!(x1, y1, z1, x2, y2, z2, psi, states)
    @cuda threads=Int(NTHR_PER_BLK) blocks=Int(NBLOCK) zero_stats_kernel!(stats)
    @cuda threads=Int(NTHR_PER_BLK) blocks=Int(NBLOCK) propagate_kernel!(NEQ, x1, y1, z1, x2, y2, z2, psi, stats, states)

    r1_tot = 0.0; r1_sq_tot = 0.0
    r2_tot = 0.0; r2_sq_tot = 0.0
    r12_tot = 0.0; r12_sq_tot = 0.0
    naccept = 0.0
    elapsed_ns = 0.0

    for _ in 1:nsample
        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=Int(NTHR_PER_BLK) blocks=Int(NBLOCK) zero_stats_kernel!(stats)
        @cuda threads=Int(NTHR_PER_BLK) blocks=Int(NBLOCK) propagate_kernel!(
            NGEN_PER_BLOCK, x1, y1, z1, x2, y2, z2, psi, stats, states)
        reduce_stats!(stats, statsum, blocksums)
        CUDA.synchronize()
        elapsed_ns += time_ns() - t0

        s = Array(statsum)
        naccept += Float64(s[4])
        r1 = Float64(s[1]) / Float64(NGEN_PER_BLOCK * NPOINT)
        r2 = Float64(s[2]) / Float64(NGEN_PER_BLOCK * NPOINT)
        r12 = Float64(s[3]) / Float64(NGEN_PER_BLOCK * NPOINT)
        r1_tot += r1; r1_sq_tot += r1 * r1
        r2_tot += r2; r2_sq_tot += r2 * r2
        r12_tot += r12; r12_sq_tot += r12 * r12
    end

    r1_tot /= nsample; r1_sq_tot /= nsample
    r2_tot /= nsample; r2_sq_tot /= nsample
    r12_tot /= nsample; r12_sq_tot /= nsample

    r1s = sqrt(max(0.0, (r1_sq_tot - r1_tot * r1_tot) / nsample))
    r2s = sqrt(max(0.0, (r2_sq_tot - r2_tot * r2_tot) / nsample))
    r12s = sqrt(max(0.0, (r12_sq_tot - r12_tot * r12_tot) / nsample))
    acceptance = 100.0 * naccept / Float64(NPOINT) / Float64(NGEN_PER_BLOCK) / Float64(nsample)

    @printf(" <r1>  = %.6f +- %.6f\n", r1_tot, r1s)
    @printf(" <r2>  = %.6f +- %.6f\n", r2_tot, r2s)
    @printf(" <r12> = %.6f +- %.6f\n", r12_tot, r12s)
    @printf(" acceptance ratio=%.1f%%\n", acceptance)
    @printf("Average execution time of kernels: %f (s)\n", (elapsed_ns * 1e-9) / nsample)

    if check
        ok = all(isfinite, (r1_tot, r2_tot, r12_tot, acceptance)) &&
             0.0 < r1_tot < 10.0 && 0.0 < r2_tot < 10.0 &&
             0.0 < r12_tot < 10.0 && 0.0 < acceptance < 100.0
        println(ok ? "PASS" : "FAIL")
        ok || exit(1)
    end
end

main()
