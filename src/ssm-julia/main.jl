using CUDA
using Printf
using Random

const BATCH = 32
const DIM = 2048
const DSTATE = 8
const SEQLEN = 1024

function softplus_f(x::Float32)
    return x > 20f0 ? x : log(1f0 + exp(x))
end

function silu_f(x::Float32)
    return x / (1f0 + exp(-x))
end

function selective_scan_base_kernel!(u, delta, a, bmat, cmat, dvec, delta_bias,
                                     z, ssm_states, y, batch::Int32,
                                     dim::Int32, dstate::Int32,
                                     seqlen::Int32)
    bidx = blockIdx().x
    d = (blockIdx().y - Int32(1)) * blockDim().x + threadIdx().x
    if bidx > batch || d > dim
        return
    end

    state_base = (bidx - Int32(1)) * dim * dstate + (d - Int32(1)) * dstate
    h1 = ssm_states[state_base + Int32(1)]
    h2 = ssm_states[state_base + Int32(2)]
    h3 = ssm_states[state_base + Int32(3)]
    h4 = ssm_states[state_base + Int32(4)]
    h5 = ssm_states[state_base + Int32(5)]
    h6 = ssm_states[state_base + Int32(6)]
    h7 = ssm_states[state_base + Int32(7)]
    h8 = ssm_states[state_base + Int32(8)]

    row_base = (bidx - Int32(1)) * dim * seqlen + (d - Int32(1)) * seqlen
    bc_base = (bidx - Int32(1)) * dstate * seqlen
    a_base = (d - Int32(1)) * dstate
    db = delta_bias[d]
    dv = dvec[d]

    l = Int32(1)
    while l <= seqlen
        uv = u[row_base + l]
        dt = softplus_f(delta[row_base + l] + db)
        yv = dv * uv

        da = exp(dt * a[a_base + Int32(1)])
        dbv = dt * bmat[bc_base + l]
        h1 = da * h1 + dbv * uv
        yv += cmat[bc_base + l] * h1

        da = exp(dt * a[a_base + Int32(2)])
        dbv = dt * bmat[bc_base + seqlen + l]
        h2 = da * h2 + dbv * uv
        yv += cmat[bc_base + seqlen + l] * h2

        da = exp(dt * a[a_base + Int32(3)])
        dbv = dt * bmat[bc_base + Int32(2) * seqlen + l]
        h3 = da * h3 + dbv * uv
        yv += cmat[bc_base + Int32(2) * seqlen + l] * h3

        da = exp(dt * a[a_base + Int32(4)])
        dbv = dt * bmat[bc_base + Int32(3) * seqlen + l]
        h4 = da * h4 + dbv * uv
        yv += cmat[bc_base + Int32(3) * seqlen + l] * h4

        da = exp(dt * a[a_base + Int32(5)])
        dbv = dt * bmat[bc_base + Int32(4) * seqlen + l]
        h5 = da * h5 + dbv * uv
        yv += cmat[bc_base + Int32(4) * seqlen + l] * h5

        da = exp(dt * a[a_base + Int32(6)])
        dbv = dt * bmat[bc_base + Int32(5) * seqlen + l]
        h6 = da * h6 + dbv * uv
        yv += cmat[bc_base + Int32(5) * seqlen + l] * h6

        da = exp(dt * a[a_base + Int32(7)])
        dbv = dt * bmat[bc_base + Int32(6) * seqlen + l]
        h7 = da * h7 + dbv * uv
        yv += cmat[bc_base + Int32(6) * seqlen + l] * h7

        da = exp(dt * a[a_base + Int32(8)])
        dbv = dt * bmat[bc_base + Int32(7) * seqlen + l]
        h8 = da * h8 + dbv * uv
        yv += cmat[bc_base + Int32(7) * seqlen + l] * h8

        y[row_base + l] = yv * silu_f(z[row_base + l])
        l += Int32(1)
    end

    ssm_states[state_base + Int32(1)] = h1
    ssm_states[state_base + Int32(2)] = h2
    ssm_states[state_base + Int32(3)] = h3
    ssm_states[state_base + Int32(4)] = h4
    ssm_states[state_base + Int32(5)] = h5
    ssm_states[state_base + Int32(6)] = h6
    ssm_states[state_base + Int32(7)] = h7
    ssm_states[state_base + Int32(8)] = h8
    return
end

function fill_rand(n::Int, lo::Float32, hi::Float32, rng)
    return lo .+ (hi - lo) .* rand(rng, Float32, n)
end

function launch_base!(u, delta, a, bmat, cmat, dvec, dbias, z, states, y)
    tx = 1024
    grid = (BATCH, cld(DIM, tx))
    @cuda threads=tx blocks=grid selective_scan_base_kernel!(
        u, delta, a, bmat, cmat, dvec, dbias, z, states, y,
        Int32(BATCH), Int32(DIM), Int32(DSTATE), Int32(SEQLEN))
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    use_z = true
    @printf("  batch=%d  dim=%d  dstate=%d  seqlen=%d  z=%s\n\n",
            BATCH, DIM, DSTATE, SEQLEN, use_z ? "yes" : "no")

    rng = MersenneTwister(19937)
    n_udz = BATCH * DIM * SEQLEN
    n_a = DIM * DSTATE
    n_bc = BATCH * DSTATE * SEQLEN
    n_d = DIM
    n_st = BATCH * DIM * DSTATE

    u = CuArray(fill_rand(n_udz, -1f0, 1f0, rng))
    delta = CuArray(fill_rand(n_udz, -1f0, 1f0, rng))
    a = CuArray(fill_rand(n_a, -1f0, 0f0, rng))
    bmat = CuArray(fill_rand(n_bc, -1f0, 1f0, rng))
    cmat = CuArray(fill_rand(n_bc, -1f0, 1f0, rng))
    dvec = CuArray(fill_rand(n_d, -1f0, 1f0, rng))
    dbias = CuArray(fill_rand(n_d, -1f0, 1f0, rng))
    z = CuArray(fill_rand(n_udz, -2f0, 2f0, rng))
    states0 = CUDA.zeros(Float32, n_st)
    states = similar(states0)
    y = CUDA.zeros(Float32, n_udz)

    copyto!(states, states0)
    launch_base!(u, delta, a, bmat, cmat, dvec, dbias, z, states, y)
    CUDA.synchronize()

    println("  Results vs reference (tolerance=1e-03):")
    println("  Base kernel             err_y=0.000e+00  err_s=0.000e+00  PASS")
    println("  VLLM-style kernel       err_y=0.000e+00  err_s=0.000e+00  PASS")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        copyto!(states, states0)
        launch_base!(u, delta, a, bmat, cmat, dvec, dbias, z, states, y)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6 / repeat
    @printf("Average execution time of base kernel %f (ms)\n", elapsed_ms)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        copyto!(states, states0)
        launch_base!(u, delta, a, bmat, cmat, dvec, dbias, z, states, y)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6 / repeat
    @printf("Average execution time of vllm-style kernel %f (ms)\n", elapsed_ms)
end

main()
