using CUDA
using Printf
using Random

function grouped_gemm_ref!(c, a, b, m, n, k)
    @inbounds for g in eachindex(m)
        Ag = a[g]
        Bg = b[g]
        Cg = c[g]
        for j in 1:n[g], i in 1:m[g]
            acc = 0.0f0
            for p in 1:k[g]
                acc += Ag[i, p] * Bg[p, j]
            end
            Cg[i, j] = acc
        end
    end
end

function verify_grouped(test, ref, tol)
    max_err = 0.0
    @inbounds for g in eachindex(ref)
        for i in eachindex(ref[g])
            max_err = max(max_err, abs(Float64(test[g][i]) - Float64(ref[g][i])))
        end
    end
    ok = max_err <= tol
    @printf("Maximum absolute error: %e (tolerance %e) -> %s\n",
            max_err, tol, ok ? "PASS" : "FAIL")
    return ok
end

function performance(m, n, k, avg_time_us)
    total_ops = 0.0
    for g in eachindex(m)
        total_ops += 2.0 * m[g] * n[g] * k[g]
    end
    perf = total_ops / (avg_time_us * 1.0e3)
    scale = "G"
    if perf >= 1000
        perf /= 1000
        scale = "T"
    end
    @printf("Average execution time: %.3f (us) | performance: %.2f %sFLOP/s\n",
            avg_time_us, perf, scale)
end

function grouped_gemm(name, m, n, k, repeat)
    println(">>>>>>>>>>>>>>> ", name, " grouped GEMM >>>>>>>>>>>>>>>")
    groups = length(m)
    rng = MersenneTwister(123)

    hA = [rand(rng, Float32, m[g], k[g]) for g in 1:groups]
    hB = [rand(rng, Float32, k[g], n[g]) for g in 1:groups]
    hRef = [zeros(Float32, m[g], n[g]) for g in 1:groups]
    dA = CuArray.(hA)
    dB = CuArray.(hB)
    dC = [CUDA.zeros(Float32, m[g], n[g]) for g in 1:groups]

    transa = fill('N', groups)
    transb = fill('N', groups)
    alpha = fill(1.0f0, groups)
    beta = fill(0.0f0, groups)

    for _ in 1:30
        CUDA.CUBLAS.gemm_grouped_batched!(transa, transb, alpha, dA, dB, beta, dC)
    end
    CUDA.synchronize()

    hC = Array.(dC)
    grouped_gemm_ref!(hRef, hA, hB, m, n, k)
    verify_grouped(hC, hRef, 1.0)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        CUDA.CUBLAS.gemm_grouped_batched!(transa, transb, alpha, dA, dB, beta, dC)
    end
    CUDA.synchronize()
    avg_us = (time_ns() - start) * 1.0e-3 / repeat
    performance(m, n, k, avg_us)
end

function main()
    repeat = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 100
    num_experts = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 64
    hidden = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 2048
    inter = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 2048
    avg_tokens = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 16

    num_tokens = num_experts * avg_tokens
    rng = MersenneTwister(123)
    cdf = Vector{Float64}(undef, num_experts)
    wsum = 0.0
    for e in 1:num_experts
        wsum += 0.2 + rand(rng)
        cdf[e] = wsum
    end

    m = zeros(Int, num_experts)
    n = fill(inter, num_experts)
    k = fill(hidden, num_experts)
    for _ in 1:num_tokens
        r = rand(rng) * wsum
        e = 1
        while e < num_experts && r > cdf[e]
            e += 1
        end
        m[e] += 1
    end
    for e in 1:num_experts
        m[e] == 0 && (m[e] = 1)
    end

    @printf("MoE FFN grouped GEMM: %d experts, hidden(K)=%d, intermediate(N)=%d, repeat=%d\n",
            num_experts, hidden, inter, repeat)
    grouped_gemm("Half precision", m, n, k, repeat)
end

main()
