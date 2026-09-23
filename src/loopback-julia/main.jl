using CUDA
using Printf
using Random

const TAUSWORTHE_NUM_THREADS = Int32(256)
const TAUSWORTHE_NUM_BLOCKS = Int32(4096)
const TAUSWORTHE_TOTAL_NUM_THREADS = TAUSWORTHE_NUM_THREADS * TAUSWORTHE_NUM_BLOCKS
const TAUSWORTHE_NUM_SEEDS_PER_GENERATOR = Int32(4)
const TAUSWORTHE_NUM_SEEDS = TAUSWORTHE_TOTAL_NUM_THREADS * TAUSWORTHE_NUM_SEEDS_PER_GENERATOR
const LOOKBACK_MAX_T = Int32((4096 - 256) ÷ 256)
const LOOKBACK_NUM_PARAMETER_VALUES = TAUSWORTHE_TOTAL_NUM_THREADS
const LOOKBACK_PATHS_PER_SIM = Int32(512)
const PI_F = Float32(pi)

@inline function taus_step(z::UInt32, s1::Int32, s2::Int32, s3::Int32, m::UInt32)
    b = ((z << s1) ⊻ z) >> s2
    nz = ((z & m) << s3) ⊻ b
    return nz
end

@inline function lcg_step(z::UInt32)
    return UInt32(1664525) * z + UInt32(1013904223)
end

@inline function uniform_taus(z1::UInt32, z2::UInt32, z3::UInt32, z4::UInt32)
    z1n = taus_step(z1, Int32(13), Int32(19), Int32(12), UInt32(4294967294))
    z2n = taus_step(z2, Int32(2), Int32(25), Int32(4), UInt32(4294967288))
    z3n = taus_step(z3, Int32(3), Int32(11), Int32(17), UInt32(4294967280))
    z4n = lcg_step(z4)
    v = 2.3283064f-10 * Float32((z1n ⊻ z2n ⊻ z3n) ⊻ z4n)
    return v, z1n, z2n, z3n, z4n
end

@inline function box_muller(u1::Float32, u2::Float32)
    z1 = sqrt(-2.0f0 * log(u1))
    a = 2.0f0 * PI_F * u2
    return z1 * sin(a), z1 * cos(a)
end

@inline function random_taus(z1::UInt32, z2::UInt32, z3::UInt32, z4::UInt32,
                             temporary::Float32, phase::UInt32)
    if (phase & UInt32(1)) != UInt32(0)
        return temporary, temporary, z1, z2, z3, z4
    end
    u1, z1, z2, z3, z4 = uniform_taus(z1, z2, z3, z4)
    u2, z1, z2, z3, z4 = uniform_taus(z1, z2, z3, z4)
    r1, r2 = box_muller(u1, u2)
    return r1, r2, z1, z2, z3, z4
end

@inline function lookback_sim(num_cycles::UInt32, vol0::Float32, eps0::Float32,
                              a0::Float32, a1::Float32, a2::Float32, s0::Float32,
                              mu::Float32, z1::UInt32, z2::UInt32, z3::UInt32,
                              z4::UInt32, path)
    temporary = 0.0f0
    vol = vol0
    eps = eps0
    s = s0
    tx = threadIdx().x
    base = tx

    for t in UInt32(0):(num_cycles - UInt32(1))
        path[base] = s
        base += TAUSWORTHE_NUM_THREADS
        vol = sqrt(a0 + a1 * vol * vol + a2 * eps * eps)
        r, temporary, z1, z2, z3, z4 = random_taus(z1, z2, z3, z4, temporary, t)
        eps = r * vol
        eps = max(min(eps, 1.0f0), -1.0f0)
        s *= exp(mu + eps)
    end

    total = 0.0f0
    for _ in UInt32(0):(num_cycles - UInt32(1))
        base -= TAUSWORTHE_NUM_THREADS
        total += max(path[base] - s, 0.0f0)
    end
    return total, z1, z2, z3, z4
end

function tausworthe_lookback!(num_cycles::UInt32, seed_values, means, variances,
                              vol0, eps0, a0, a1, a2, s0, mu)
    path = CUDA.@cuStaticSharedMem(Float32, 256 * 15)
    address0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    address = address0 + Int32(1)

    z1 = seed_values[address]
    z2 = seed_values[address + TAUSWORTHE_TOTAL_NUM_THREADS]
    z3 = seed_values[address + Int32(2) * TAUSWORTHE_TOTAL_NUM_THREADS]
    z4 = seed_values[address + Int32(3) * TAUSWORTHE_TOTAL_NUM_THREADS]

    mean = 0.0f0
    variance = 0.0f0
    for i in UInt32(1):UInt32(LOOKBACK_PATHS_PER_SIM)
        res, z1, z2, z3, z4 = lookback_sim(
            num_cycles, vol0[address], eps0[address], a0[address], a1[address],
            a2[address], s0[address], mu[address], z1, z2, z3, z4, path)
        delta = res - mean
        mean += delta / Float32(i)
        variance += delta * (res - mean)
    end

    means[address] = mean
    variances[address] = variance / Float32(LOOKBACK_PATHS_PER_SIM - Int32(1))
    return
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <dump> <repeat>")
        exit(1)
    end
    dump = parse(Int, ARGS[1])
    repeat = parse(Int, ARGS[2])

    n = Int(LOOKBACK_NUM_PARAMETER_VALUES)
    rng = MersenneTwister(1)
    vol0 = rand(rng, Float32, n)
    a0 = rand(rng, Float32, n)
    a1 = rand(rng, Float32, n)
    a2 = rand(rng, Float32, n)
    s0 = rand(rng, Float32, n)
    eps0 = rand(rng, Float32, n)
    mu = rand(rng, Float32, n)
    seeds = rand(rng, UInt32, Int(TAUSWORTHE_NUM_SEEDS)) .+ UInt32(16)

    d_vol0 = CuArray(vol0)
    d_a0 = CuArray(a0)
    d_a1 = CuArray(a1)
    d_a2 = CuArray(a2)
    d_s0 = CuArray(s0)
    d_eps0 = CuArray(eps0)
    d_mu = CuArray(mu)
    d_seeds = CuArray(seeds)
    d_means = CUDA.zeros(Float32, n)
    d_variances = CUDA.zeros(Float32, n)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=Int(TAUSWORTHE_NUM_THREADS) blocks=Int(TAUSWORTHE_NUM_BLOCKS) tausworthe_lookback!(
            UInt32(LOOKBACK_MAX_T), d_seeds, d_means, d_variances, d_vol0, d_eps0,
            d_a0, d_a1, d_a2, d_s0, d_mu)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - start) * 1.0e-9 / repeat
    @printf("Average kernel execution time %f (s)\n", elapsed_s)

    if dump != 0
        means = Array(d_means)
        variances = Array(d_variances)
        for i in 1:n
            @printf("%d %.3f %.3f\n", i - 1, means[i], variances[i])
        end
    end
end

main()
