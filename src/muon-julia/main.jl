using CUDA
using LinearAlgebra
using Printf
using Random

const NS_A = 3.4445f0
const NS_B = -4.7750f0
const NS_C = 2.0315f0
const NS_STEPS = 5

function muon_step!(p, mom, g, beta::Float32, lr::Float32, wd::Float32)
    update = (1f0 - beta) .* g .+ beta .* (beta .* mom .+ (1f0 - beta) .* g)
    mom .= beta .* mom .+ (1f0 - beta) .* g

    P, Q = size(p)
    scale = sqrt(max(1f0, Float32(P) / Float32(Q)))
    X = P > Q ? copy(transpose(update)) : copy(update)
    X ./= (sqrt(sum(abs2, X)) + 1f-7)

    for _ in 1:NS_STEPS
        A = X * transpose(X)
        A2 = A * A
        B = NS_B .* A .+ NS_C .* A2
        X .= NS_A .* X .+ B * X
    end

    update2 = P > Q ? copy(transpose(X)) : X
    p .= p .* (1f0 - lr * wd) .- lr .* (scale .* update2)
    return p
end

function init_arrays(P::Int, Q::Int)
    rng = MersenneTwister(19937)
    mom = rand(rng, Float32, P, Q) .* 2f0 .- 1f0
    g = rand(rng, Float32, P, Q) .* 2f0 .- 1f0
    p = rand(rng, Float32, P, Q) .* 2f0 .- 1f0
    return p, mom, g
end

function run_gpu(P::Int, Q::Int, repeat::Int)
    p, mom, g = init_arrays(P, Q)
    d_p = CuArray(p)
    d_m = CuArray(mom)
    d_g = CuArray(g)
    beta = 0.95f0
    lr = 0.02f0
    wd = 0f0

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        muon_step!(d_p, d_m, d_g, beta, lr, wd)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average step execution time %f (ms)\n", elapsed * 1e-6 / repeat)
    return Array(d_p)
end

function run_reference(P::Int, Q::Int, repeat::Int)
    p, mom, g = init_arrays(P, Q)
    beta = 0.95f0
    lr = 0.02f0
    wd = 0f0
    for _ in 1:repeat
        muon_step!(p, mom, g, beta, lr, wd)
    end
    return p
end

function main()
    if length(ARGS) != 4
        @printf("Usage: %s <rows> <cols> <repeat> <verify>\n", PROGRAM_FILE)
        println("  rows = out_features, cols = in_features of a 2D weight matrix")
        return 1
    end

    P = parse(Int, ARGS[1])
    Q = parse(Int, ARGS[2])
    repeat = parse(Int, ARGS[3])
    verify = parse(Int, ARGS[4])

    p = run_gpu(P, Q, repeat)
    if verify != 0
        r = run_reference(P, Q, repeat)
        ok = maximum(abs.(r .- p)) <= 1f-2
        println(ok ? "PASS" : "FAIL")
        return ok ? 0 : 1
    end
    return 0
end

exit(main())
