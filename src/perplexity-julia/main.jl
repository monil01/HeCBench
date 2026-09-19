using CUDA
using Printf
using Random

function sigmas_kernel!(distances, p_out, desired_entropy::Float32,
                        epochs::Int32, tol::Float32, n::Int32, k::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i >= n
        return
    end

    beta_min = -Inf32
    beta_max = Inf32
    beta = 1.0f0
    ik = i * k

    for _ in Int32(0):epochs-Int32(1)
        sum_pi = eps(Float32)

        for j in Int32(0):k-Int32(1)
            idx = ik + j + Int32(1)
            @inbounds v = CUDA.exp(-distances[idx] * beta)
            @inbounds p_out[idx] = v
            sum_pi += v
        end

        sum_disti_pi = 0.0f0
        div = 1.0f0 / sum_pi
        for j in Int32(0):k-Int32(1)
            idx = ik + j + Int32(1)
            @inbounds p_out[idx] *= div
            @inbounds sum_disti_pi += distances[idx] * p_out[idx]
        end

        entropy = CUDA.log(sum_pi) + beta * sum_disti_pi
        entropy_diff = entropy - desired_entropy
        if abs(entropy_diff) <= tol
            break
        end

        if entropy_diff > 0.0f0
            beta_min = beta
            beta = beta_max == Inf32 ? beta * 2.0f0 : (beta + beta_max) * 0.5f0
        else
            beta_max = beta
            beta = beta_min == -Inf32 ? beta * 0.5f0 : (beta + beta_min) * 0.5f0
        end
    end
    return
end

function reference!(distances::Vector{Float32}, p_out::Vector{Float32},
                    perplexity::Int, epochs::Int, tol::Float32,
                    n::Int, k::Int)
    desired_entropy = log(Float32(perplexity))
    for i in 0:n-1
        beta_min = -Inf32
        beta_max = Inf32
        beta = 1.0f0
        ik = i * k
        for _ in 1:epochs
            sum_pi = eps(Float32)
            for j in 0:k-1
                idx = ik + j + 1
                v = exp(-distances[idx] * beta)
                p_out[idx] = v
                sum_pi += v
            end
            sum_disti_pi = 0.0f0
            div = 1.0f0 / sum_pi
            for j in 0:k-1
                idx = ik + j + 1
                p_out[idx] *= div
                sum_disti_pi += distances[idx] * p_out[idx]
            end
            entropy = log(sum_pi) + beta * sum_disti_pi
            entropy_diff = entropy - desired_entropy
            if abs(entropy_diff) <= tol
                break
            end
            if entropy_diff > 0.0f0
                beta_min = beta
                beta = isinf(beta_max) ? beta * 2.0f0 : (beta + beta_max) * 0.5f0
            else
                beta_max = beta
                beta = isinf(beta_min) ? beta * 0.5f0 : (beta + beta_min) * 0.5f0
            end
        end
    end
end

function run_perplexity(n::Int, perplexity::Int, repeat::Int)
    n_nbrs = 4 * perplexity
    max_iter = 100
    tol = 1.0f-8
    total = n * n_nbrs

    rng = MersenneTwister(123)
    distances = rand(rng, Float32, total)
    h_data = Vector{Float32}(undef, total)
    h_ref = Vector{Float32}(undef, total)

    d_distance = CuArray(distances)
    d_data = CUDA.zeros(Float32, total)
    desired_entropy = log(Float32(perplexity))

    elapsed_ns = 0
    for _ in 1:repeat
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=256 blocks=cld(n, 256) sigmas_kernel!(
            d_distance, d_data, desired_entropy, Int32(max_iter), tol, Int32(n), Int32(n_nbrs))
        CUDA.synchronize()
        elapsed_ns += time_ns() - start
    end
    @printf("Average kernel execution time: %f (s)\n", elapsed_ns * 1e-9 / repeat)

    copyto!(h_data, d_data)
    reference!(distances, h_ref, perplexity, max_iter, tol, n, n_nbrs)

    ok = true
    for i in eachindex(h_data)
        if abs(h_data[i] - h_ref[i]) > 1.0f-3
            @printf("%d %f %f\n", i - 1, h_data[i], h_ref[i])
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <number of points> <perplexity> <repeat>")
        return 1
    end
    return run_perplexity(parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]))
end

exit(main())
